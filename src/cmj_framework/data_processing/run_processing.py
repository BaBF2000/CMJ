import os
import json
import traceback
import datetime
from typing import Dict, List, Tuple, Any, Callable, Optional

from cmj_framework.data_processing.run_error_handler import move_to_rejected
from cmj_framework.data_processing.run_processing_temp_data import TempProcessedData
from cmj_framework.utils.json_manipulation import load_json
from cmj_framework.utils.metrics import JumpMetrics
from cmj_framework.utils.pathmanager import PathManager
from cmj_framework.utils.roi import CMJ_ROI
from cmj_framework.utils.validation import validate_trial_auto, log_validation


ProgressCB = Callable[[int, int], None]   # (i, total)
LogCB = Callable[[str], None]


def build_trial_name_from_json_path(json_path: str) -> str:
    """
    Build canonical trial name from extracted CMJ JSON filename.

    Example
    -------
    1_05112025_03_data_cmj.json -> Trial_03
    """
    stem = os.path.splitext(os.path.basename(json_path))[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    return "Trial_" + parts[2]


def extract_session_key_from_filename(json_path: str) -> Optional[str]:
    """
    Example: 1_05112025_03_data_cmj.json -> "1_05112025"
    Rule: take the first 2 chunks separated by "_".
    """
    name = os.path.splitext(os.path.basename(json_path))[0]
    parts = name.split("_")
    if len(parts) < 2:
        return None
    return "{}_{}".format(parts[0], parts[1])


def filter_json_by_session_key(
    json_files: List[str],
    log_cb: Optional[LogCB] = None,
) -> Tuple[List[str], Optional[str]]:
    """Keep only files matching the same session key as the first valid file."""
    if not json_files:
        return [], None

    ref_key = None
    for path in json_files:
        key = extract_session_key_from_filename(path)
        if key:
            ref_key = key
            break

    if not ref_key:
        if log_cb:
            log_cb("✖ Sitzungsschlüssel konnte aus den Dateinamen nicht bestimmt werden.")
        return json_files, None

    kept, ignored = [], []
    for path in json_files:
        key = extract_session_key_from_filename(path)
        (kept if key == ref_key else ignored).append(path)

    if ignored and log_cb:
        log_cb("⚠ {} JSON(s) ignoriert (anderer Sitzungsschlüssel als '{}'):".format(len(ignored), ref_key))
        for path in ignored:
            log_cb("   - {}".format(os.path.basename(path)))

    return kept, ref_key



def write_combined_metrics_json(
    json_files: List[str],
    results: Dict[str, Dict[str, Any]],
    session_key: Optional[str],
    log_cb: Optional[LogCB] = None,
) -> None:
    """
    Save a combined metrics JSON in each session processed folder.
    Grouping is performed by PathManager resolved from each file's user_info.
    """
    grouped: Dict[str, List[str]] = {}
    pm_by_group: Dict[str, PathManager] = {}

    for path in json_files:
        pm = PathManager.from_extracted_json(path)
        group_key = "{}::{}".format(pm.patient_name, pm.session_date)
        grouped.setdefault(group_key, []).append(path)
        pm_by_group[group_key] = pm

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for group_key, files in grouped.items():
        pm = pm_by_group[group_key]

        session_results: Dict[str, Dict[str, Any]] = {}
        for path in files:
            trial_name = build_trial_name_from_json_path(path)
            if trial_name in results:
                session_results[trial_name] = results[trial_name]

        if not session_results:
            continue

        out_name = "{}_combined.json".format(session_key) if session_key else "combined_metrics.json"
        out_path = pm.processed_file(out_name)

        payload = {
            "generated_at": timestamp,
            "patient_name": pm.patient_name,
            "session_date": pm.session_date,
            "trial_count": len(session_results),
            "metrics_by_trial": session_results,
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")

        if log_cb:
            log_cb("✅ Kombinierte Metriken gespeichert: {}".format(out_path))


def process_multiple_json(
    json_files: List[str],
    progress_cb: Optional[ProgressCB] = None,
    log_cb: Optional[LogCB] = None,
    save_combined: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Process multiple extracted JSON files and compute metrics."""
    results: Dict[str, Dict[str, Any]] = {}
    logs: List[str] = []

    def _log(*parts) -> None:
        message = " ".join(str(part) for part in parts)
        logs.append(message)
        if log_cb:
            log_cb(message)

    if not json_files:
        _log("Keine Eingabedateien.")
        return results, logs

    json_files, session_key = filter_json_by_session_key(json_files, log_cb=log_cb)
    total = len(json_files)

    for i, json_path in enumerate(json_files, start=1):
        json_path = os.path.abspath(json_path)
        trial_name = build_trial_name_from_json_path(json_path)

        try:
            if not os.path.exists(json_path):
                raise FileNotFoundError("File not found: {}".format(json_path))

            trial = TempProcessedData.load(json_path, trial_name, log_cb=_log)

            metrics_obj = JumpMetrics(
                trial.Fz_l,
                trial.Fz_r,
                trial.F_total,
                trial.trajectory,
                trial.plate_rate,
            )

            roi_total = CMJ_ROI(
                trial.F_total,
                trial.trajectory,
                trial.plate_rate,
            )

            validation_result = validate_trial_auto(
                roi_T=roi_total,
                metrics=metrics_obj,
                total_force=trial.F_total,
                rate=trial.plate_rate,
            )

            trial.validation_result = validation_result

            try:
                pm = PathManager.from_extracted_json(json_path)
                log_validation(
                    pm=pm,
                    trial=trial.trial_name,
                    validation_result=validation_result,
                )
            except Exception as log_err:
                _log("{} ⚠ Validierungs-Log konnte nicht geschrieben werden: {}".format(trial_name, log_err))

            metrics_payload = dict(metrics_obj.all_metrics)
            metrics_payload["Validation"] = validation_result

            results[trial.trial_name] = metrics_payload

            _log("{} ✔ Verarbeitet".format(trial_name))
            _log(
                "{} Validation: {}{}".format(
                    trial_name,
                    validation_result.get("status", "UNKNOWN"),
                    (
                        " | " + "; ".join(validation_result.get("reasons", []))
                        if validation_result.get("reasons")
                        else ""
                    ),
                )
            )

        except Exception as exc:
            _log("{} ✖ {}".format(trial_name, exc))
            _log(traceback.format_exc())

            try:
                move_to_rejected(json_path, error_message=str(exc))
            except Exception as move_err:
                _log("{} ✖ Verschieben nach 'rejected' fehlgeschlagen: {}".format(trial_name, move_err))
                _log(traceback.format_exc())

        finally:
            if progress_cb:
                progress_cb(i, total)

    if save_combined and results:
        try:
            write_combined_metrics_json(
                json_files=json_files,
                results=results,
                session_key=session_key,
                log_cb=log_cb,
            )
        except Exception as export_err:
            _log("✖ Kombinierte Metriken konnten nicht gespeichert werden: {}".format(export_err))
            _log(traceback.format_exc())

    return results, logs


def find_cmj_session_dir_from_path(json_path: str) -> Optional[str]:
    """
    Backward-compatible helper used by GUI modules.

    Returns the CMJ session directory for a given extracted JSON file.
    The session directory is resolved from JSON content (user_info),
    using the canonical PathManager.
    """
    try:
        pm = PathManager.from_extracted_json(json_path)
        return pm.session_dir
    except Exception:
        return None