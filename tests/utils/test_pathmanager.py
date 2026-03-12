import json
from pathlib import Path

import pytest

from cmj_framework.utils.pathmanager import PathManager


def test_sanitize_replaces_forbidden_characters():
    """
    Test that forbidden filesystem characters are replaced with underscores.
    """
    raw_name = 'Max:/\\*?"<>| Mustermann'
    sanitized_name = PathManager.sanitize(raw_name)

    assert sanitized_name == "Max_________ Mustermann"


def test_canonical_base_dir_uses_documents_folder(monkeypatch):
    """
    Test that the canonical base directory is built from the user home directory.
    """
    monkeypatch.setattr("os.path.expanduser", lambda _: "/fake/home")

    base_dir = PathManager.canonical_base_dir()

    assert Path(base_dir) == Path("/fake/home") / "Documents" / "CMJ_manager"


def test_init_creates_expected_directory_structure(tmp_path, monkeypatch):
    """
    Test that PathManager creates the expected directory structure.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    manager = PathManager(patient_name="Max Mustermann", session_date="01.01.2020")

    assert Path(manager.base_dir) == tmp_path
    assert Path(manager.patient_dir) == tmp_path / "Max Mustermann"
    assert Path(manager.session_dir) == tmp_path / "Max Mustermann" / "01.01.2020"
    assert Path(manager.raw_dir) == tmp_path / "Max Mustermann" / "01.01.2020" / "raw_data"
    assert Path(manager.processed_dir) == tmp_path / "Max Mustermann" / "01.01.2020" / "processed"
    assert Path(manager.reports_dir) == tmp_path / "Max Mustermann" / "01.01.2020" / "reports"
    assert Path(manager.rejected_dir) == tmp_path / "Max Mustermann" / "01.01.2020" / "rejected"

    assert Path(manager.patient_dir).exists()
    assert Path(manager.session_dir).exists()
    assert Path(manager.raw_dir).exists()
    assert Path(manager.processed_dir).exists()
    assert Path(manager.reports_dir).exists()
    assert Path(manager.rejected_dir).exists()


def test_init_uses_current_date_when_session_date_is_none(tmp_path, monkeypatch):
    """
    Test that the current date is used when no session date is provided.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    manager = PathManager(patient_name="Max Mustermann")

    assert isinstance(manager.session_date, str)
    assert len(manager.session_date) == 10


def test_summary_returns_expected_keys(tmp_path, monkeypatch):
    """
    Test that the summary dictionary contains all expected keys.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    manager = PathManager(patient_name="Max Mustermann", session_date="01.01.2020")
    summary = manager.summary()

    expected_keys = {
        "base",
        "patient",
        "session",
        "raw",
        "processed",
        "reports",
        "rejected",
    }

    assert set(summary.keys()) == expected_keys


def test_raw_file_returns_expected_path(tmp_path, monkeypatch):
    """
    Test that raw_file returns the expected path inside raw_data.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    manager = PathManager(patient_name="Max Mustermann", session_date="01.01.2020")
    result = manager.raw_file("trial_01.json")

    assert Path(result) == Path(manager.raw_dir) / "trial_01.json"


def test_processed_file_returns_expected_path(tmp_path, monkeypatch):
    """
    Test that processed_file returns the expected path inside processed.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    manager = PathManager(patient_name="Max Mustermann", session_date="01.01.2020")
    result = manager.processed_file("processed_01.json")

    assert Path(result) == Path(manager.processed_dir) / "processed_01.json"


def test_report_file_returns_expected_path(tmp_path, monkeypatch):
    """
    Test that report_file returns the expected path inside reports.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    manager = PathManager(patient_name="Max Mustermann", session_date="01.01.2020")
    result = manager.report_file("report.docx")

    assert Path(result) == Path(manager.reports_dir) / "report.docx"


def test_rejected_file_returns_expected_path(tmp_path, monkeypatch):
    """
    Test that rejected_file returns the expected path inside rejected.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    manager = PathManager(patient_name="Max Mustermann", session_date="01.01.2020")
    result = manager.rejected_file("bad_trial.json")

    assert Path(result) == Path(manager.rejected_dir) / "bad_trial.json"


def test_from_extracted_json_reads_user_info(tmp_path, monkeypatch):
    """
    Test that PathManager.from_extracted_json uses user_info values from JSON.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    json_path = tmp_path / "trial.json"
    json_data = {
        "user_info": {
            "name": "Max Mustermann",
            "trial_date": "01.01.2020",
        }
    }
    json_path.write_text(json.dumps(json_data), encoding="utf-8")

    manager = PathManager.from_extracted_json(str(json_path))

    assert manager.patient_name == "Max Mustermann"
    assert manager.session_date == "01.01.2020"


def test_from_extracted_json_uses_folder_fallback_when_user_info_missing(tmp_path, monkeypatch):
    """
    Test that PathManager.from_extracted_json falls back to the folder structure
    when user_info is missing.
    """
    monkeypatch.setattr(PathManager, "get_base_dir", lambda self: str(tmp_path / "output"))
    monkeypatch.setattr(PathManager, "install_demo_content_if_missing", lambda self: None)

    json_path = (
        tmp_path
        / "Max Mustermann"
        / "01.01.2020"
        / "raw_data"
        / "trial.json"
    )
    json_path.parent.mkdir(parents=True)
    json_path.write_text("{}", encoding="utf-8")

    manager = PathManager.from_extracted_json(str(json_path))

    assert manager.patient_name == "Max Mustermann"
    assert manager.session_date != ""


def test_install_demo_content_to_base_dir_copies_demo_if_missing(tmp_path, monkeypatch):
    """
    Test that demo content is copied to the base directory if it is missing.
    """
    demo_root = tmp_path / "demo_source"
    demo_patient_dir = demo_root / "Max, Mustermann"
    demo_patient_dir.mkdir(parents=True)
    (demo_patient_dir / "example.txt").write_text("demo", encoding="utf-8")

    monkeypatch.setattr(PathManager, "demo_source_dir", classmethod(lambda cls: demo_patient_dir))

    base_dir = tmp_path / "base_dir"
    base_dir.mkdir()

    PathManager.install_demo_content_to_base_dir(str(base_dir))

    copied_demo_dir = base_dir / "Max, Mustermann"
    assert copied_demo_dir.exists()
    assert (copied_demo_dir / "example.txt").exists()


def test_install_demo_content_to_base_dir_does_nothing_if_demo_missing(tmp_path, monkeypatch):
    """
    Test that no exception is raised if the demo source directory does not exist.
    """
    missing_demo_dir = tmp_path / "missing_demo"

    monkeypatch.setattr(PathManager, "demo_source_dir", classmethod(lambda cls: missing_demo_dir))

    base_dir = tmp_path / "base_dir"
    base_dir.mkdir()

    PathManager.install_demo_content_to_base_dir(str(base_dir))

    assert list(base_dir.iterdir()) == []