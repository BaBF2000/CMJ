import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cmj_framework.gui import trial_review


class FakeLabel:
    def __init__(self):
        self.text_value = ""
        self.properties = {}
        self._style = SimpleNamespace(unpolish=lambda widget: None, polish=lambda widget: None)

    def setProperty(self, key, value):
        self.properties[key] = value

    def style(self):
        return self._style

    def setText(self, text):
        self.text_value = text


class FakeReviewBox:
    def __init__(self):
        self.visible = False

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False


class FakeCombo:
    def __init__(self, text=""):
        self._text = text

    def currentText(self):
        return self._text


class FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, value):
        self.emitted.append(value)


class FakeHost(trial_review.TrialReviewMixin):
    def __init__(self, current_trial="Trial_01"):
        self.status_label = FakeLabel()
        self.review_text = FakeLabel()
        self.review_box = FakeReviewBox()
        self.trial_combo = FakeCombo(current_trial)
        self.decisions = {}
        self.trialRejected = FakeSignal()
        self.refresh_called = 0

    def refresh_trials(self):
        self.refresh_called += 1


def test_find_latest_combined_in_session_returns_none_for_missing_session(tmp_path):
    """
    Test that find_latest_combined_in_session returns None for a missing session folder.
    """
    result = trial_review.find_latest_combined_in_session(str(tmp_path / "missing"))

    assert result is None


def test_find_latest_combined_in_session_returns_none_when_processed_missing(tmp_path):
    """
    Test that find_latest_combined_in_session returns None when the processed folder is missing.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    result = trial_review.find_latest_combined_in_session(str(session_dir))

    assert result is None


def test_find_latest_combined_in_session_returns_latest_file(tmp_path):
    """
    Test that find_latest_combined_in_session returns the newest *_combined.json file.
    """
    session_dir = tmp_path / "session"
    processed_dir = session_dir / "processed"
    processed_dir.mkdir(parents=True)

    file_old = processed_dir / "old_combined.json"
    file_new = processed_dir / "new_combined.json"

    file_old.write_text("{}", encoding="utf-8")
    file_new.write_text("{}", encoding="utf-8")

    file_old.touch()
    file_new.touch()

    result = trial_review.find_latest_combined_in_session(str(session_dir))

    assert result in {file_old, file_new}


def test_remove_trial_from_combined_path_returns_false_when_file_missing(tmp_path):
    """
    Test that remove_trial_from_combined_path returns False when the file does not exist.
    """
    result = trial_review.remove_trial_from_combined_path(tmp_path / "missing.json", "Trial_01")

    assert result is False


def test_remove_trial_from_combined_path_returns_false_when_metrics_invalid(tmp_path):
    """
    Test that remove_trial_from_combined_path returns False when metrics_by_trial is not a dict.
    """
    combined_path = tmp_path / "combined.json"
    combined_path.write_text(
        json.dumps({"metrics_by_trial": []}),
        encoding="utf-8",
    )

    result = trial_review.remove_trial_from_combined_path(combined_path, "Trial_01")

    assert result is False


def test_remove_trial_from_combined_path_returns_false_when_trial_missing(tmp_path):
    """
    Test that remove_trial_from_combined_path returns False when the trial is not in metrics_by_trial.
    """
    combined_path = tmp_path / "combined.json"
    combined_path.write_text(
        json.dumps({"metrics_by_trial": {"Trial_02": {}}, "trial_count": 1}),
        encoding="utf-8",
    )

    result = trial_review.remove_trial_from_combined_path(combined_path, "Trial_01")

    assert result is False


def test_remove_trial_from_combined_path_removes_trial_and_updates_count(tmp_path):
    """
    Test that remove_trial_from_combined_path removes the trial and updates trial_count.
    """
    combined_path = tmp_path / "combined.json"
    combined_path.write_text(
        json.dumps(
            {
                "metrics_by_trial": {
                    "Trial_01": {"jump_height": 20.0},
                    "Trial_02": {"jump_height": 22.0},
                },
                "trial_count": 2,
            }
        ),
        encoding="utf-8",
    )

    result = trial_review.remove_trial_from_combined_path(combined_path, "Trial_01")

    assert result is True

    payload = json.loads(combined_path.read_text(encoding="utf-8"))
    assert payload["trial_count"] == 1
    assert "Trial_01" not in payload["metrics_by_trial"]
    assert "Trial_02" in payload["metrics_by_trial"]

    backup_path = combined_path.with_suffix(".json.bak")
    assert backup_path.exists()


def test_set_status_pending_updates_label():
    """
    Test that _set_status('pending') updates the label text and state.
    """
    host = FakeHost()

    host._set_status("pending")

    assert host.status_label.properties["state"] == "pending"
    assert "Entscheidung ausstehend" in host.status_label.text_value


def test_set_status_keep_updates_label():
    """
    Test that _set_status('keep') updates the label text and state.
    """
    host = FakeHost()

    host._set_status("keep")

    assert host.status_label.properties["state"] == "keep"
    assert "Behalten" in host.status_label.text_value


def test_set_status_reject_updates_label():
    """
    Test that _set_status('reject') updates the label text and state.
    """
    host = FakeHost()

    host._set_status("reject")

    assert host.status_label.properties["state"] == "reject"
    assert "Abgelehnt" in host.status_label.text_value


def test_show_review_panel_sets_pending_state_and_shows_panel():
    """
    Test that _show_review_panel prepares the UI for review.
    """
    host = FakeHost("Trial_03")

    host._show_review_panel("Trial_03")

    assert host.review_box.visible is True
    assert "Trial: Trial_03" in host.review_text.text_value
    assert host.status_label.properties["state"] == "pending"


def test_later_current_trial_sets_decision_and_hides_panel():
    """
    Test that _later_current_trial stores the decision and hides the panel.
    """
    host = FakeHost("Trial_03")
    host.review_box.show()

    host._later_current_trial()

    assert host.decisions["Trial_03"] == "later"
    assert host.review_box.visible is False
    assert host.status_label.properties["state"] == "pending"


def test_keep_current_trial_sets_decision_and_hides_panel():
    """
    Test that _keep_current_trial stores the decision and hides the panel.
    """
    host = FakeHost("Trial_03")
    host.review_box.show()

    host._keep_current_trial()

    assert host.decisions["Trial_03"] == "keep"
    assert host.review_box.visible is False
    assert host.status_label.properties["state"] == "keep"


def test_reject_current_trial_sets_decision_and_calls_reject(monkeypatch):
    """
    Test that _reject_current_trial stores the decision and calls reject_trial.
    """
    host = FakeHost("Trial_03")
    host.review_box.show()

    called = {"trial": None}
    monkeypatch.setattr(host, "reject_trial", lambda trial_name: called.__setitem__("trial", trial_name))

    host._reject_current_trial()

    assert host.decisions["Trial_03"] == "reject"
    assert host.review_box.visible is False
    assert host.status_label.properties["state"] == "reject"
    assert called["trial"] == "Trial_03"


def test_reject_trial_warns_when_no_json_path(monkeypatch):
    """
    Test that reject_trial warns when no JSON path is available.
    """
    host = FakeHost("Trial_03")

    monkeypatch.setattr(trial_review.TempProcessedData, "get_trial", lambda name: SimpleNamespace(json_path=None))

    warnings = []
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    host.reject_trial("Trial_03")

    assert len(warnings) == 1
    assert "Kein Dateipfad" in warnings[0][2]


def test_reject_trial_warns_when_file_missing(monkeypatch):
    """
    Test that reject_trial warns when the JSON file does not exist.
    """
    host = FakeHost("Trial_03")

    monkeypatch.setattr(
        trial_review.TempProcessedData,
        "get_trial",
        lambda name: SimpleNamespace(json_path="C:/missing/file.json"),
    )
    monkeypatch.setattr(trial_review.os.path, "exists", lambda path: False)

    warnings = []
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    host.reject_trial("Trial_03")

    assert len(warnings) == 1
    assert "Datei nicht gefunden" in warnings[0][2]


def test_reject_trial_executes_full_success_flow(monkeypatch, tmp_path):
    """
    Test that reject_trial performs the full happy-path rejection flow.
    """
    host = FakeHost("Trial_03")

    json_path = tmp_path / "trial.json"
    json_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        trial_review.TempProcessedData,
        "get_trial",
        lambda name: SimpleNamespace(json_path=str(json_path)),
    )
    monkeypatch.setattr(trial_review.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        trial_review,
        "find_cmj_session_dir_from_path",
        lambda path: str(tmp_path / "session"),
    )
    monkeypatch.setattr(
        trial_review,
        "find_latest_combined_in_session",
        lambda session_dir: tmp_path / "session" / "processed" / "latest_combined.json",
    )
    monkeypatch.setattr(
        trial_review,
        "remove_trial_from_combined_path",
        lambda combined_path, trial_name: True,
    )
    monkeypatch.setattr(
        trial_review.PathManager,
        "from_extracted_json",
        lambda path: SimpleNamespace(patient_name="Max", session_date="01.01.2020"),
    )
    monkeypatch.setattr(
        trial_review,
        "invalidate_trial_manual",
        lambda reason: {"status": "INVALID_MANUAL", "reasons": [reason]},
    )

    log_calls = []
    monkeypatch.setattr(trial_review, "log_validation", lambda **kwargs: log_calls.append(kwargs))

    move_calls = []
    monkeypatch.setattr(
        trial_review,
        "move_to_rejected",
        lambda json_path, error_message: move_calls.append((json_path, error_message)),
    )

    removed_trials = []
    monkeypatch.setattr(
        trial_review.TempProcessedData,
        "remove_trial",
        lambda trial_name: removed_trials.append(trial_name),
    )

    infos = []
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "information",
        lambda *args: infos.append(args),
    )
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "warning",
        lambda *args: None,
    )
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "critical",
        lambda *args: None,
    )

    host.reject_trial("Trial_03")

    assert len(log_calls) == 1
    assert len(move_calls) == 1
    assert move_calls[0][1] == "Rejected by user (GUI)."
    assert removed_trials == ["Trial_03"]
    assert host.trialRejected.emitted == [str(json_path.resolve())]
    assert host.refresh_called == 1
    assert len(infos) == 1
    assert "combined-Datei wurde aktualisiert" in infos[0][2]


def test_reject_trial_shows_warning_when_combined_update_fails(monkeypatch, tmp_path):
    """
    Test that reject_trial shows a warning if removing from combined JSON fails.
    """
    host = FakeHost("Trial_03")

    json_path = tmp_path / "trial.json"
    json_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        trial_review.TempProcessedData,
        "get_trial",
        lambda name: SimpleNamespace(json_path=str(json_path)),
    )
    monkeypatch.setattr(trial_review.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        trial_review,
        "find_cmj_session_dir_from_path",
        lambda path: str(tmp_path / "session"),
    )
    monkeypatch.setattr(
        trial_review,
        "find_latest_combined_in_session",
        lambda session_dir: tmp_path / "session" / "processed" / "latest_combined.json",
    )

    def raise_combined(*args, **kwargs):
        raise RuntimeError("combined update failed")

    monkeypatch.setattr(trial_review, "remove_trial_from_combined_path", raise_combined)
    monkeypatch.setattr(
        trial_review.PathManager,
        "from_extracted_json",
        lambda path: SimpleNamespace(patient_name="Max", session_date="01.01.2020"),
    )
    monkeypatch.setattr(
        trial_review,
        "invalidate_trial_manual",
        lambda reason: {"status": "INVALID_MANUAL", "reasons": [reason]},
    )
    monkeypatch.setattr(trial_review, "log_validation", lambda **kwargs: None)
    monkeypatch.setattr(trial_review, "move_to_rejected", lambda *args, **kwargs: None)
    monkeypatch.setattr(trial_review.TempProcessedData, "remove_trial", lambda trial_name: None)

    warnings = []
    infos = []

    monkeypatch.setattr(
        trial_review.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "information",
        lambda *args: infos.append(args),
    )
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "critical",
        lambda *args: None,
    )

    host.reject_trial("Trial_03")

    assert len(warnings) == 1
    assert "combined-Datei entfernen" in warnings[0][2]
    assert len(infos) == 1


def test_reject_trial_shows_critical_when_move_to_rejected_fails(monkeypatch, tmp_path):
    """
    Test that reject_trial shows a critical error and stops if move_to_rejected fails.
    """
    host = FakeHost("Trial_03")

    json_path = tmp_path / "trial.json"
    json_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        trial_review.TempProcessedData,
        "get_trial",
        lambda name: SimpleNamespace(json_path=str(json_path)),
    )
    monkeypatch.setattr(trial_review.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        trial_review,
        "find_cmj_session_dir_from_path",
        lambda path: None,
    )
    monkeypatch.setattr(
        trial_review.PathManager,
        "from_extracted_json",
        lambda path: SimpleNamespace(patient_name="Max", session_date="01.01.2020"),
    )
    monkeypatch.setattr(
        trial_review,
        "invalidate_trial_manual",
        lambda reason: {"status": "INVALID_MANUAL", "reasons": [reason]},
    )
    monkeypatch.setattr(trial_review, "log_validation", lambda **kwargs: None)

    def raise_move(*args, **kwargs):
        raise RuntimeError("cannot move")

    monkeypatch.setattr(trial_review, "move_to_rejected", raise_move)

    criticals = []
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "critical",
        lambda *args: criticals.append(args),
    )
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "warning",
        lambda *args: None,
    )
    monkeypatch.setattr(
        trial_review.QMessageBox,
        "information",
        lambda *args: None,
    )

    host.reject_trial("Trial_03")

    assert len(criticals) == 1
    assert "Verschieben nach rejected fehlgeschlagen" in criticals[0][2]
    assert host.trialRejected.emitted == []
    assert host.refresh_called == 0