import os
import shutil
import datetime
from typing import Optional, Dict, Any

from cmj_framework.utils.app_logging import get_log_file, append_to_path
from cmj_framework.utils.json_manipulation import load_json
from cmj_framework.utils.pathmanager import PathManager


gui_log_callback = None


def register_gui_logger(callback):
    """Register a GUI callback to display log lines."""
    global gui_log_callback
    gui_log_callback = callback


def _safe_read_user_info(json_path: str) -> Dict[str, Any]:
    """Read user_info from a JSON file if possible."""
    try:
        data = load_json(json_path)
        return data.get("user_info", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_path_manager(
    json_path: str,
    patient_name: Optional[str] = None,
    session_date: Optional[str] = None,
) -> PathManager:
    """
    Resolve PathManager using the central JSON-based helper,
    optionally overriding patient and/or session.
    """
    pm = PathManager.from_extracted_json(json_path)

    if patient_name or session_date:
        return PathManager(
            patient_name=patient_name or pm.patient_name,
            session_date=session_date or pm.session_date,
        )

    return pm


def write_log(pm: PathManager, filename: str, error_message: str) -> None:
    """Append a structured processing error block to the central log file."""
    del pm  # kept for API compatibility

    log_path = get_log_file("processing_log.txt")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"[{timestamp}] DATEI: {filename}\n"
        f"FEHLER: {error_message}\n"
        f"{'-' * 60}\n"
    )

    append_to_path(log_path, entry)

    if gui_log_callback:
        gui_log_callback(entry)


def move_to_rejected(
    json_path: str,
    error_message: str,
    patient_name: Optional[str] = None,
    session_date: Optional[str] = None,
) -> None:
    """
    Rename file to *_rejected.json and move it into the session rejected folder.
    Uses PathManager as the canonical path source.
    """
    pm = _resolve_path_manager(
        json_path,
        patient_name=patient_name,
        session_date=session_date,
    )

    filename = os.path.basename(json_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_rejected{ext}"

    destination = pm.rejected_file(new_filename)

    try:
        shutil.move(json_path, destination)
    except Exception as move_error:
        write_log(pm, filename, f"Fehler beim Verschieben der Datei: {move_error}")

    write_log(pm, new_filename, error_message)