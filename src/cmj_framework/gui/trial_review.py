import json
import os
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QMessageBox

from cmj_framework.data_processing.run_error_handler import move_to_rejected
from cmj_framework.data_processing.run_processing import find_cmj_session_dir_from_path
from cmj_framework.data_processing.run_processing_temp_data import TempProcessedData

from cmj_framework.utils.pathmanager import PathManager
from cmj_framework.utils.validation import invalidate_trial_manual, log_validation


def find_latest_combined_in_session(session_dir: str) -> Path | None:
    """Return the most recent *_combined.json from the processed folder of a session."""
    if not session_dir:
        return None

    processed_dir = Path(session_dir) / "processed"
    if not processed_dir.is_dir():
        return None

    candidates = list(processed_dir.glob("*_combined.json"))
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def remove_trial_from_combined_path(combined_path: str | Path, trial_name: str) -> bool:
    """
    Remove one trial from a specific combined JSON file.

    Returns
    -------
    bool
        True if the combined file was modified, False otherwise.
    """
    combined_path = Path(combined_path)

    if not combined_path.exists():
        return False

    backup_path = combined_path.with_suffix(combined_path.suffix + ".bak")

    try:
        if not backup_path.exists():
            backup_path.write_bytes(combined_path.read_bytes())
    except Exception:
        pass

    with open(combined_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    metrics = payload.get("metrics_by_trial", {})

    if not isinstance(metrics, dict):
        return False

    if trial_name not in metrics:
        return False

    del metrics[trial_name]
    payload["metrics_by_trial"] = metrics
    payload["trial_count"] = len(metrics)

    tmp_path = combined_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")

    os.replace(tmp_path, combined_path)

    return True


class TrialReviewMixin:
    """
    Reusable review and rejection logic for the CMJ viewer widget.
    """

    # ---- attributes expected from host widget (for Pylance) ----
    status_label: Any
    review_text: Any
    review_box: Any
    trial_combo: Any
    decisions: dict[str, str]
    trialRejected: Any

    def refresh_trials(self) -> None: ...

    # ------------------------------------------------------------

    def _set_status(self, state: str) -> None:
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

        if state == "keep":
            self.status_label.setText("✅ Behalten")
        elif state == "reject":
            self.status_label.setText("❌ Abgelehnt")
        else:
            self.status_label.setText("⏳ Entscheidung ausstehend")

    def _show_review_panel(self, trial_name: str) -> None:
        if not trial_name:
            return

        self._set_status("pending")

        self.review_text.setText(
            f"Möchten Sie diesen Versuch behalten?\n\nTrial: {trial_name}"
        )

        self.review_box.show()

    def _later_current_trial(self) -> None:
        name = self.trial_combo.currentText()

        if not name:
            return

        self.decisions[name] = "later"

        self._set_status("pending")
        self.review_box.hide()

    def _keep_current_trial(self) -> None:
        name = self.trial_combo.currentText()

        if not name:
            return

        self.decisions[name] = "keep"

        self._set_status("keep")
        self.review_box.hide()

    def _reject_current_trial(self) -> None:
        name = self.trial_combo.currentText()

        if not name:
            return

        self.decisions[name] = "reject"

        self._set_status("reject")

        self.review_box.hide()

        self.reject_trial(name)

    def reject_trial(self, trial_name: str) -> None:
        """
        Reject current trial:

        1) update latest combined JSON of the session
        2) log manual validation
        3) move raw trial JSON to rejected
        4) remove trial from TempProcessedData
        5) refresh viewer
        """

        trial = TempProcessedData.get_trial(trial_name)

        json_path = getattr(trial, "json_path", None) if trial else None

        if not json_path:
            QMessageBox.warning(
                self,
                "Fehler",
                "Kein Dateipfad für diesen Versuch gefunden.",
            )
            return

        json_path = os.path.abspath(json_path)

        if not os.path.exists(json_path):
            QMessageBox.warning(
                self,
                "Fehler",
                f"Datei nicht gefunden:\n{json_path}",
            )
            return

        # ---------------- session detection ----------------

        session_dir = find_cmj_session_dir_from_path(json_path)

        combined_path = (
            find_latest_combined_in_session(session_dir)
            if session_dir
            else None
        )

        combined_updated = False

        if combined_path is not None:
            try:
                combined_updated = remove_trial_from_combined_path(
                    combined_path,
                    trial_name,
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Warnung",
                    f"Konnte den Versuch nicht aus der combined-Datei entfernen:\n{exc}",
                )

        # ---------------- manual validation log ----------------

        try:
            pm = PathManager.from_extracted_json(json_path)

            manual_validation = invalidate_trial_manual(
                "Rejected by user (GUI)."
            )

            log_validation(
                pm=pm,
                trial=trial_name,
                validation_result=manual_validation,
            )

        except Exception:
            pass

        # ---------------- move file to rejected ----------------

        try:
            move_to_rejected(
                json_path,
                error_message="Rejected by user (GUI).",
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Fehler",
                f"Verschieben nach rejected fehlgeschlagen:\n{exc}",
            )

            return

        # ---------------- remove from memory ----------------

        if hasattr(TempProcessedData, "remove_trial"):
            TempProcessedData.remove_trial(trial_name)

        # ---------------- update viewer ----------------

        self.trialRejected.emit(json_path)

        self.refresh_trials()

        # ---------------- UI feedback ----------------

        if combined_updated:

            QMessageBox.information(
                self,
                "Erledigt",
                "Versuch wurde abgelehnt.\n"
                "Die combined-Datei wurde aktualisiert.",
            )

        else:

            QMessageBox.information(
                self,
                "Erledigt",
                "Versuch wurde abgelehnt und entfernt.",
            )