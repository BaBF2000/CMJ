from pathlib import Path
from types import SimpleNamespace

import pytest

from cmj_framework.data_processing import run_error_handler


@pytest.fixture(autouse=True)
def reset_gui_logger():
    """
    Reset the global GUI logger before and after each test.
    """
    run_error_handler.gui_log_callback = None
    yield
    run_error_handler.gui_log_callback = None


def test_register_gui_logger_sets_global_callback():
    """
    Test that register_gui_logger stores the callback globally.
    """
    def callback(message):
        return message

    run_error_handler.register_gui_logger(callback)

    assert run_error_handler.gui_log_callback is callback


def test_safe_read_user_info_returns_user_info_when_json_is_valid(monkeypatch):
    """
    Test that _safe_read_user_info returns the user_info dict from a valid JSON payload.
    """
    monkeypatch.setattr(
        run_error_handler,
        "load_json",
        lambda path: {"user_info": {"name": "Max", "trial_date": "01.01.2020"}},
    )

    result = run_error_handler._safe_read_user_info("fake.json")

    assert result == {"name": "Max", "trial_date": "01.01.2020"}


def test_safe_read_user_info_returns_empty_dict_when_json_is_not_dict(monkeypatch):
    """
    Test that _safe_read_user_info returns an empty dict if the JSON root is not a dict.
    """
    monkeypatch.setattr(run_error_handler, "load_json", lambda path: ["not", "a", "dict"])

    result = run_error_handler._safe_read_user_info("fake.json")

    assert result == {}


def test_safe_read_user_info_returns_empty_dict_on_error(monkeypatch):
    """
    Test that _safe_read_user_info returns an empty dict when JSON loading fails.
    """
    def raise_error(path):
        raise RuntimeError("cannot read")

    monkeypatch.setattr(run_error_handler, "load_json", raise_error)

    result = run_error_handler._safe_read_user_info("fake.json")

    assert result == {}


def test_resolve_path_manager_returns_from_extracted_json_when_no_override(monkeypatch):
    """
    Test that _resolve_path_manager returns PathManager.from_extracted_json when no override is given.
    """
    fake_pm = SimpleNamespace(patient_name="Max", session_date="01.01.2020")

    monkeypatch.setattr(
        run_error_handler.PathManager,
        "from_extracted_json",
        lambda path: fake_pm,
    )

    result = run_error_handler._resolve_path_manager("fake.json")

    assert result is fake_pm


def test_resolve_path_manager_uses_override_values(monkeypatch):
    """
    Test that _resolve_path_manager builds a new PathManager when overrides are provided.
    """
    base_pm = SimpleNamespace(patient_name="Max", session_date="01.01.2020")

    created = {}

    class FakePathManager:
        @staticmethod
        def from_extracted_json(path):
            return base_pm

        def __init__(self, patient_name, session_date):
            created["patient_name"] = patient_name
            created["session_date"] = session_date
            self.patient_name = patient_name
            self.session_date = session_date

    monkeypatch.setattr(run_error_handler, "PathManager", FakePathManager)

    result = run_error_handler._resolve_path_manager(
        "fake.json",
        patient_name="Override Patient",
        session_date="02.02.2020",
    )

    assert created == {
        "patient_name": "Override Patient",
        "session_date": "02.02.2020",
    }
    assert result.patient_name == "Override Patient"
    assert result.session_date == "02.02.2020"


def test_resolve_path_manager_uses_partial_override(monkeypatch):
    """
    Test that _resolve_path_manager keeps non-overridden values from the extracted PathManager.
    """
    base_pm = SimpleNamespace(patient_name="Max", session_date="01.01.2020")

    created = {}

    class FakePathManager:
        @staticmethod
        def from_extracted_json(path):
            return base_pm

        def __init__(self, patient_name, session_date):
            created["patient_name"] = patient_name
            created["session_date"] = session_date
            self.patient_name = patient_name
            self.session_date = session_date

    monkeypatch.setattr(run_error_handler, "PathManager", FakePathManager)

    result = run_error_handler._resolve_path_manager(
        "fake.json",
        patient_name="Override Patient",
        session_date=None,
    )

    assert created == {
        "patient_name": "Override Patient",
        "session_date": "01.01.2020",
    }
    assert result.patient_name == "Override Patient"
    assert result.session_date == "01.01.2020"


def test_write_log_appends_message_to_processing_log(monkeypatch, tmp_path):
    """
    Test that write_log appends a structured log entry to the processing log file.
    """
    log_file = tmp_path / "processing_log.txt"
    captured = {}

    monkeypatch.setattr(run_error_handler, "get_log_file", lambda filename: log_file)

    def fake_append_to_path(path, message):
        captured["path"] = path
        captured["message"] = message

    monkeypatch.setattr(run_error_handler, "append_to_path", fake_append_to_path)

    pm = SimpleNamespace()
    run_error_handler.write_log(pm, "trial.json", "Something went wrong")

    assert captured["path"] == log_file
    assert "DATEI: trial.json" in captured["message"]
    assert "FEHLER: Something went wrong" in captured["message"]


def test_write_log_calls_gui_logger_when_registered(monkeypatch, tmp_path):
    """
    Test that write_log forwards the log entry to the GUI callback when registered.
    """
    log_file = tmp_path / "processing_log.txt"
    gui_messages = []

    monkeypatch.setattr(run_error_handler, "get_log_file", lambda filename: log_file)
    monkeypatch.setattr(run_error_handler, "append_to_path", lambda path, message: path)

    run_error_handler.register_gui_logger(gui_messages.append)

    pm = SimpleNamespace()
    run_error_handler.write_log(pm, "trial.json", "GUI error")

    assert len(gui_messages) == 1
    assert "DATEI: trial.json" in gui_messages[0]
    assert "FEHLER: GUI error" in gui_messages[0]


def test_move_to_rejected_moves_file_and_logs_error(monkeypatch, tmp_path):
    """
    Test that move_to_rejected renames the file, moves it to the rejected folder,
    and writes the final error log.
    """
    json_file = tmp_path / "trial.json"
    json_file.write_text("{}", encoding="utf-8")

    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()

    destination = rejected_dir / "trial_rejected.json"

    fake_pm = SimpleNamespace(
        rejected_file=lambda filename: str(rejected_dir / filename),
    )

    monkeypatch.setattr(run_error_handler, "_resolve_path_manager", lambda *args, **kwargs: fake_pm)

    move_calls = []
    monkeypatch.setattr(
        run_error_handler.shutil,
        "move",
        lambda src, dst: move_calls.append((src, dst)),
    )

    log_calls = []

    def fake_write_log(pm, filename, error_message):
        log_calls.append((filename, error_message))

    monkeypatch.setattr(run_error_handler, "write_log", fake_write_log)

    run_error_handler.move_to_rejected(str(json_file), "broken trial")

    assert move_calls == [(str(json_file), str(destination))]
    assert log_calls == [("trial_rejected.json", "broken trial")]


def test_move_to_rejected_logs_move_error_and_final_error(monkeypatch, tmp_path):
    """
    Test that move_to_rejected logs both the move failure and the original error.
    """
    json_file = tmp_path / "trial.json"
    json_file.write_text("{}", encoding="utf-8")

    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()

    fake_pm = SimpleNamespace(
        rejected_file=lambda filename: str(rejected_dir / filename),
    )

    monkeypatch.setattr(run_error_handler, "_resolve_path_manager", lambda *args, **kwargs: fake_pm)

    def raise_move(src, dst):
        raise OSError("cannot move file")

    monkeypatch.setattr(run_error_handler.shutil, "move", raise_move)

    log_calls = []

    def fake_write_log(pm, filename, error_message):
        log_calls.append((filename, error_message))

    monkeypatch.setattr(run_error_handler, "write_log", fake_write_log)

    run_error_handler.move_to_rejected(str(json_file), "broken trial")

    assert len(log_calls) == 2
    assert log_calls[0][0] == "trial.json"
    assert "Fehler beim Verschieben der Datei" in log_calls[0][1]
    assert log_calls[1] == ("trial_rejected.json", "broken trial")


def test_move_to_rejected_passes_overrides_to_resolve_path_manager(monkeypatch, tmp_path):
    """
    Test that move_to_rejected forwards patient/session overrides to _resolve_path_manager.
    """
    json_file = tmp_path / "trial.json"
    json_file.write_text("{}", encoding="utf-8")

    captured = {}

    fake_pm = SimpleNamespace(
        rejected_file=lambda filename: str(tmp_path / filename),
    )

    def fake_resolve(json_path, patient_name=None, session_date=None):
        captured["json_path"] = json_path
        captured["patient_name"] = patient_name
        captured["session_date"] = session_date
        return fake_pm

    monkeypatch.setattr(run_error_handler, "_resolve_path_manager", fake_resolve)
    monkeypatch.setattr(run_error_handler.shutil, "move", lambda src, dst: None)
    monkeypatch.setattr(run_error_handler, "write_log", lambda pm, filename, error_message: None)

    run_error_handler.move_to_rejected(
        str(json_file),
        "broken trial",
        patient_name="Override Patient",
        session_date="02.02.2020",
    )

    assert captured == {
        "json_path": str(json_file),
        "patient_name": "Override Patient",
        "session_date": "02.02.2020",
    }