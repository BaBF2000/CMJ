import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cmj_framework.data_processing import run_processing


def test_build_trial_name_from_json_path_returns_canonical_name():
    """
    Test that the trial name is built from the third filename chunk.
    """
    result = run_processing.build_trial_name_from_json_path("1_05112025_03_data_cmj.json")

    assert result == "Trial_03"


def test_build_trial_name_from_json_path_returns_stem_if_format_is_unexpected():
    """
    Test that the filename stem is returned if the expected pattern is missing.
    """
    result = run_processing.build_trial_name_from_json_path("unexpected.json")

    assert result == "unexpected"


def test_extract_session_key_from_filename_returns_expected_key():
    """
    Test that the session key is extracted from the first two filename chunks.
    """
    result = run_processing.extract_session_key_from_filename("1_05112025_03_data_cmj.json")

    assert result == "1_05112025"


def test_extract_session_key_from_filename_returns_none_for_short_name():
    """
    Test that no session key is returned for filenames with too few chunks.
    """
    result = run_processing.extract_session_key_from_filename("singlechunk.json")

    assert result is None


def test_filter_json_by_session_key_keeps_only_matching_files():
    """
    Test that files from other sessions are ignored.
    """
    messages = []

    files = [
        "/tmp/1_05112025_01_data_cmj.json",
        "/tmp/1_05112025_02_data_cmj.json",
        "/tmp/2_05112025_01_data_cmj.json",
    ]

    kept, session_key = run_processing.filter_json_by_session_key(files, log_cb=messages.append)

    assert session_key == "1_05112025"
    assert kept == [
        "/tmp/1_05112025_01_data_cmj.json",
        "/tmp/1_05112025_02_data_cmj.json",
    ]
    assert any("ignoriert" in message for message in messages)


def test_filter_json_by_session_key_returns_input_when_no_valid_key_found():
    """
    Test that the original list is returned if no session key can be determined.
    """
    messages = []
    files = ["/tmp/invalid.json", "/tmp/another.json"]

    kept, session_key = run_processing.filter_json_by_session_key(files, log_cb=messages.append)

    assert kept == files
    assert session_key is None
    assert any("Sitzungsschlüssel" in message for message in messages)


def test_write_combined_metrics_json_writes_grouped_output(tmp_path):
    """
    Test that write_combined_metrics_json writes one combined JSON per patient/session group.
    """
    processed_dir = tmp_path / "Max Mustermann" / "01.01.2020" / "processed"
    processed_dir.mkdir(parents=True)

    file_1 = tmp_path / "a" / "1_05112025_01_data_cmj.json"
    file_2 = tmp_path / "b" / "1_05112025_02_data_cmj.json"
    file_1.parent.mkdir(parents=True)
    file_2.parent.mkdir(parents=True)
    file_1.write_text("{}", encoding="utf-8")
    file_2.write_text("{}", encoding="utf-8")

    results = {
        "Trial_01": {"jump_height": 20.0},
        "Trial_02": {"jump_height": 22.0},
    }

    class FakePathManager:
        def __init__(self):
            self.patient_name = "Max Mustermann"
            self.session_date = "01.01.2020"

        def processed_file(self, filename):
            return str(processed_dir / filename)

    fake_pm = FakePathManager()

    original = run_processing.PathManager.from_extracted_json
    run_processing.PathManager.from_extracted_json = lambda path: fake_pm
    try:
        run_processing.write_combined_metrics_json(
            json_files=[str(file_1), str(file_2)],
            results=results,
            session_key="1_05112025",
            log_cb=None,
        )
    finally:
        run_processing.PathManager.from_extracted_json = original

    output_path = processed_dir / "1_05112025_combined.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["patient_name"] == "Max Mustermann"
    assert payload["session_date"] == "01.01.2020"
    assert payload["trial_count"] == 2
    assert set(payload["metrics_by_trial"].keys()) == {"Trial_01", "Trial_02"}


def test_process_multiple_json_returns_empty_results_for_no_input():
    """
    Test that process_multiple_json handles an empty input list.
    """
    results, logs = run_processing.process_multiple_json([])

    assert results == {}
    assert any("Keine Eingabedateien." in msg for msg in logs)


def test_process_multiple_json_processes_one_valid_trial(monkeypatch):
    """
    Test that process_multiple_json processes one valid trial and stores metrics.
    """
    json_file = "/tmp/1_05112025_03_data_cmj.json"
    progress_calls = []
    logged_messages = []

    fake_trial = SimpleNamespace(
        trial_name="Trial_03",
        Fz_l=[1, 2],
        Fz_r=[3, 4],
        F_total=[4, 6],
        trajectory=[7, 8],
        plate_rate=1000.0,
        validation_result=None,
    )

    fake_metrics_obj = SimpleNamespace(
        all_metrics={"jump_height": 25.0},
    )

    monkeypatch.setattr(run_processing.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        run_processing,
        "filter_json_by_session_key",
        lambda files, log_cb=None: (files, "1_05112025"),
    )
    monkeypatch.setattr(
        run_processing.TempProcessedData,
        "load",
        lambda json_path, trial_name, log_cb=None: fake_trial,
    )
    monkeypatch.setattr(run_processing, "JumpMetrics", lambda *args, **kwargs: fake_metrics_obj)
    monkeypatch.setattr(run_processing, "CMJ_ROI", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        run_processing,
        "validate_trial_auto",
        lambda **kwargs: {"status": "VALID", "reasons": []},
    )
    monkeypatch.setattr(
        run_processing.PathManager,
        "from_extracted_json",
        lambda path: SimpleNamespace(patient_name="Max", session_date="01.01.2020"),
    )

    log_validation_calls = {"count": 0}

    def fake_log_validation(**kwargs):
        log_validation_calls["count"] += 1

    monkeypatch.setattr(run_processing, "log_validation", fake_log_validation)

    write_combined_calls = {"count": 0}

    def fake_write_combined_metrics_json(**kwargs):
        write_combined_calls["count"] += 1

    monkeypatch.setattr(run_processing, "write_combined_metrics_json", fake_write_combined_metrics_json)

    results, logs = run_processing.process_multiple_json(
        [json_file],
        progress_cb=lambda i, total: progress_calls.append((i, total)),
        log_cb=logged_messages.append,
        save_combined=True,
    )

    assert "Trial_03" in results
    assert results["Trial_03"]["jump_height"] == 25.0
    assert results["Trial_03"]["Validation"] == {"status": "VALID", "reasons": []}
    assert fake_trial.validation_result == {"status": "VALID", "reasons": []}
    assert progress_calls == [(1, 1)]
    assert log_validation_calls["count"] == 1
    assert write_combined_calls["count"] == 1
    assert any("✔ Verarbeitet" in msg for msg in logs)


def test_process_multiple_json_moves_failed_trial_to_rejected(monkeypatch):
    """
    Test that process_multiple_json moves a failed trial to rejected.
    """
    json_file = "/tmp/1_05112025_03_data_cmj.json"
    move_calls = []

    monkeypatch.setattr(run_processing.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        run_processing,
        "filter_json_by_session_key",
        lambda files, log_cb=None: (files, "1_05112025"),
    )

    def raise_on_load(json_path, trial_name, log_cb=None):
        raise RuntimeError("broken trial")

    monkeypatch.setattr(run_processing.TempProcessedData, "load", raise_on_load)
    monkeypatch.setattr(
        run_processing,
        "move_to_rejected",
        lambda json_path, error_message: move_calls.append((json_path, error_message)),
    )

    results, logs = run_processing.process_multiple_json([json_file], save_combined=False)

    assert results == {}
    assert len(move_calls) == 1
    assert move_calls[0][0] == run_processing.os.path.abspath(json_file)
    assert move_calls[0][1] == "broken trial"
    assert any("✖ broken trial" in msg for msg in logs)


def test_process_multiple_json_logs_if_validation_log_fails(monkeypatch):
    """
    Test that validation log write failures are reported but do not stop processing.
    """
    json_file = "/tmp/1_05112025_03_data_cmj.json"

    fake_trial = SimpleNamespace(
        trial_name="Trial_03",
        Fz_l=[1, 2],
        Fz_r=[3, 4],
        F_total=[4, 6],
        trajectory=[7, 8],
        plate_rate=1000.0,
        validation_result=None,
    )

    monkeypatch.setattr(run_processing.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        run_processing,
        "filter_json_by_session_key",
        lambda files, log_cb=None: (files, "1_05112025"),
    )
    monkeypatch.setattr(
        run_processing.TempProcessedData,
        "load",
        lambda json_path, trial_name, log_cb=None: fake_trial,
    )
    monkeypatch.setattr(run_processing, "JumpMetrics", lambda *args, **kwargs: SimpleNamespace(all_metrics={}))
    monkeypatch.setattr(run_processing, "CMJ_ROI", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        run_processing,
        "validate_trial_auto",
        lambda **kwargs: {"status": "VALID", "reasons": []},
    )
    monkeypatch.setattr(
        run_processing.PathManager,
        "from_extracted_json",
        lambda path: SimpleNamespace(patient_name="Max", session_date="01.01.2020"),
    )

    def raise_log_validation(**kwargs):
        raise RuntimeError("cannot write validation log")

    monkeypatch.setattr(run_processing, "log_validation", raise_log_validation)

    results, logs = run_processing.process_multiple_json([json_file], save_combined=False)

    assert "Trial_03" in results
    assert any("Validierungs-Log konnte nicht geschrieben werden" in msg for msg in logs)


def test_process_multiple_json_logs_if_combined_export_fails(monkeypatch):
    """
    Test that combined export failures are logged.
    """
    json_file = "/tmp/1_05112025_03_data_cmj.json"

    fake_trial = SimpleNamespace(
        trial_name="Trial_03",
        Fz_l=[1, 2],
        Fz_r=[3, 4],
        F_total=[4, 6],
        trajectory=[7, 8],
        plate_rate=1000.0,
        validation_result=None,
    )

    monkeypatch.setattr(run_processing.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        run_processing,
        "filter_json_by_session_key",
        lambda files, log_cb=None: (files, "1_05112025"),
    )
    monkeypatch.setattr(
        run_processing.TempProcessedData,
        "load",
        lambda json_path, trial_name, log_cb=None: fake_trial,
    )
    monkeypatch.setattr(run_processing, "JumpMetrics", lambda *args, **kwargs: SimpleNamespace(all_metrics={}))
    monkeypatch.setattr(run_processing, "CMJ_ROI", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        run_processing,
        "validate_trial_auto",
        lambda **kwargs: {"status": "VALID", "reasons": []},
    )
    monkeypatch.setattr(
        run_processing.PathManager,
        "from_extracted_json",
        lambda path: SimpleNamespace(patient_name="Max", session_date="01.01.2020"),
    )
    monkeypatch.setattr(run_processing, "log_validation", lambda **kwargs: None)

    def raise_combined_export(**kwargs):
        raise RuntimeError("combined export failed")

    monkeypatch.setattr(run_processing, "write_combined_metrics_json", raise_combined_export)

    results, logs = run_processing.process_multiple_json([json_file], save_combined=True)

    assert "Trial_03" in results
    assert any("Kombinierte Metriken konnten nicht gespeichert werden" in msg for msg in logs)


def test_find_cmj_session_dir_from_path_returns_session_dir(monkeypatch):
    """
    Test that find_cmj_session_dir_from_path returns the resolved session directory.
    """
    monkeypatch.setattr(
        run_processing.PathManager,
        "from_extracted_json",
        lambda path: SimpleNamespace(session_dir="/tmp/Max/01.01.2020"),
    )

    result = run_processing.find_cmj_session_dir_from_path("/tmp/file.json")

    assert result == "/tmp/Max/01.01.2020"


def test_find_cmj_session_dir_from_path_returns_none_on_error(monkeypatch):
    """
    Test that find_cmj_session_dir_from_path returns None if resolution fails.
    """
    def raise_pm(path):
        raise RuntimeError("cannot resolve")

    monkeypatch.setattr(run_processing.PathManager, "from_extracted_json", raise_pm)

    result = run_processing.find_cmj_session_dir_from_path("/tmp/file.json")

    assert result is None