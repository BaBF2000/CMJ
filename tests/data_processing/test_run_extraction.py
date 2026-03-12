import json
import importlib
from pathlib import Path

import pytest

run_extraction = importlib.import_module("cmj_framework.data_processing.run_extraction")

def test_parse_nexus_version_from_path_returns_expected_tuple():
    """
    Test that parse_nexus_version_from_path extracts the version correctly.
    """
    result = run_extraction.parse_nexus_version_from_path(
        r"C:\Program Files (x86)\Vicon\Nexus2.12.3\Python\python.exe"
    )

    assert result == (2, 12, 3)


def test_parse_nexus_version_from_path_pads_missing_parts():
    """
    Test that parse_nexus_version_from_path pads missing version parts with zeros.
    """
    result = run_extraction.parse_nexus_version_from_path(
        r"C:\Program Files (x86)\Vicon\Nexus2.12\Python\python.exe"
    )

    assert result == (2, 12, 0)


def test_parse_nexus_version_from_path_returns_zero_tuple_when_missing():
    """
    Test that parse_nexus_version_from_path returns zeros if no Nexus part exists.
    """
    result = run_extraction.parse_nexus_version_from_path(
        r"C:\Program Files\Python\python.exe"
    )

    assert result == (0, 0, 0)


def test_ensure_app_config_exists_returns_existing_file(tmp_path, monkeypatch):
    """
    Test that ensure_app_config_exists returns the existing config file.
    """
    cfg_path = tmp_path / "vicon_path_config.json"
    cfg_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_extraction, "app_config_path", lambda: cfg_path)

    result = run_extraction.ensure_app_config_exists()

    assert result == cfg_path


def test_ensure_app_config_exists_copies_default_file(tmp_path, monkeypatch):
    """
    Test that ensure_app_config_exists copies the default config when needed.
    """
    cfg_path = tmp_path / "vicon_path_config.json"
    default_path = tmp_path / "vicon_path_config.default.json"
    default_content = {"vicon": {"nexus_python27_path": "C:/Vicon/python.exe"}}
    default_path.write_text(json.dumps(default_content), encoding="utf-8")

    monkeypatch.setattr(run_extraction, "app_config_path", lambda: cfg_path)
    monkeypatch.setattr(run_extraction, "default_app_config_path", lambda: default_path)

    result = run_extraction.ensure_app_config_exists()

    assert result == cfg_path
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == default_content


def test_ensure_app_config_exists_creates_fallback_when_default_missing(tmp_path, monkeypatch):
    """
    Test that ensure_app_config_exists creates fallback content when no default exists.
    """
    cfg_path = tmp_path / "vicon_path_config.json"
    default_path = tmp_path / "missing.default.json"

    monkeypatch.setattr(run_extraction, "app_config_path", lambda: cfg_path)
    monkeypatch.setattr(run_extraction, "default_app_config_path", lambda: default_path)

    result = run_extraction.ensure_app_config_exists()

    assert result == cfg_path
    content = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "vicon" in content
    assert "nexus_python27_path" in content["vicon"]
    assert "extract_script" in content["vicon"]


def test_load_app_config_returns_parsed_json(tmp_path, monkeypatch):
    """
    Test that load_app_config returns the parsed config dictionary.
    """
    cfg_path = tmp_path / "vicon_path_config.json"
    expected = {"vicon": {"extract_script": "script.py"}}
    cfg_path.write_text(json.dumps(expected), encoding="utf-8")

    monkeypatch.setattr(run_extraction, "ensure_app_config_exists", lambda: cfg_path)

    result = run_extraction.load_app_config()

    assert result == expected


def test_load_app_config_returns_empty_dict_on_invalid_json(tmp_path, monkeypatch):
    """
    Test that load_app_config returns an empty dict when the config is invalid.
    """
    cfg_path = tmp_path / "vicon_path_config.json"
    cfg_path.write_text("{invalid json}", encoding="utf-8")

    monkeypatch.setattr(run_extraction, "ensure_app_config_exists", lambda: cfg_path)

    result = run_extraction.load_app_config()

    assert result == {}


def test_get_vicon_config_returns_vicon_section(monkeypatch):
    """
    Test that get_vicon_config returns the vicon section when available.
    """
    monkeypatch.setattr(
        run_extraction,
        "load_app_config",
        lambda: {"vicon": {"nexus_python27_path": "C:/Vicon/python.exe"}},
    )

    result = run_extraction.get_vicon_config()

    assert result == {"nexus_python27_path": "C:/Vicon/python.exe"}


def test_get_vicon_config_returns_empty_dict_for_missing_section(monkeypatch):
    """
    Test that get_vicon_config returns an empty dict when the vicon section is missing.
    """
    monkeypatch.setattr(run_extraction, "load_app_config", lambda: {})

    result = run_extraction.get_vicon_config()

    assert result == {}


def test_find_nexus_python27_prefers_configured_path(monkeypatch):
    """
    Test that find_nexus_python27 prefers the configured path when it exists.
    """
    configured = r"C:\Vicon\Nexus2.12\Python\python.exe"

    monkeypatch.setattr(
        run_extraction,
        "get_vicon_config",
        lambda: {"nexus_python27_path": configured},
    )
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: path == configured)

    result = run_extraction.find_nexus_python27()

    assert result == configured


def test_find_nexus_python27_uses_environment_override(monkeypatch):
    """
    Test that find_nexus_python27 uses the environment variable when the configured path is absent.
    """
    env_path = r"C:\Env\Vicon\Nexus2.12\Python\python.exe"

    monkeypatch.setattr(run_extraction, "get_vicon_config", lambda: {})
    monkeypatch.setattr(run_extraction.os, "environ", {"CMJ_NEXUS_PY27": env_path})
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: path == env_path)

    result = run_extraction.find_nexus_python27()

    assert result == env_path


def test_find_nexus_python27_selects_highest_version_from_scan(monkeypatch):
    """
    Test that find_nexus_python27 returns the highest version found during scanning.
    """
    candidates = [
        r"C:\Program Files (x86)\Vicon\Nexus2.10\Python\python.exe",
        r"C:\Program Files (x86)\Vicon\Nexus2.12\Python\python.exe",
        r"C:\Program Files (x86)\Vicon\Nexus2.9\Python\python.exe",
    ]

    monkeypatch.setattr(
        run_extraction,
        "get_vicon_config",
        lambda: {"nexus_search_roots": [r"C:\Program Files (x86)\Vicon"]},
    )
    monkeypatch.setattr(run_extraction.os, "environ", {})
    monkeypatch.setattr(run_extraction.glob, "glob", lambda pattern: list(candidates))
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: path in candidates)

    result = run_extraction.find_nexus_python27()

    assert result == r"C:\Program Files (x86)\Vicon\Nexus2.12\Python\python.exe"


def test_find_nexus_python27_uses_default_path_as_last_fallback(monkeypatch):
    """
    Test that find_nexus_python27 falls back to DEFAULT_NEXUS_PY27 if it exists.
    """
    monkeypatch.setattr(run_extraction, "get_vicon_config", lambda: {})
    monkeypatch.setattr(run_extraction.os, "environ", {})
    monkeypatch.setattr(run_extraction.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(
        run_extraction.os.path,
        "exists",
        lambda path: path == run_extraction.DEFAULT_NEXUS_PY27,
    )

    result = run_extraction.find_nexus_python27()

    assert result == run_extraction.DEFAULT_NEXUS_PY27


def test_find_nexus_python27_returns_none_when_nothing_is_found(monkeypatch):
    """
    Test that find_nexus_python27 returns None if no interpreter is found.
    """
    monkeypatch.setattr(run_extraction, "get_vicon_config", lambda: {})
    monkeypatch.setattr(run_extraction.os, "environ", {})
    monkeypatch.setattr(run_extraction.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: False)

    result = run_extraction.find_nexus_python27()

    assert result is None


def test_validate_nexus_python27_path_rejects_empty_path():
    """
    Test that validate_nexus_python27_path rejects an empty path.
    """
    valid, message = run_extraction.validate_nexus_python27_path("")

    assert valid is False
    assert "Kein Pfad eingetragen" in message


def test_validate_nexus_python27_path_rejects_missing_file(monkeypatch):
    """
    Test that validate_nexus_python27_path rejects a missing file.
    """
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: False)

    valid, message = run_extraction.validate_nexus_python27_path(r"C:\missing\python.exe")

    assert valid is False
    assert "Datei nicht gefunden" in message


def test_validate_nexus_python27_path_rejects_non_file(monkeypatch):
    """
    Test that validate_nexus_python27_path rejects a path that is not a file.
    """
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: True)
    monkeypatch.setattr(run_extraction.os.path, "isfile", lambda path: False)

    valid, message = run_extraction.validate_nexus_python27_path(r"C:\folder")

    assert valid is False
    assert "Pfad ist keine Datei" in message


def test_validate_nexus_python27_path_rejects_non_python_executable(monkeypatch):
    """
    Test that validate_nexus_python27_path rejects a file with an unexpected name.
    """
    path = r"C:\Vicon\Nexus2.12\Python\not_python.exe"

    monkeypatch.setattr(run_extraction.os.path, "exists", lambda p: True)
    monkeypatch.setattr(run_extraction.os.path, "isfile", lambda p: True)

    valid, message = run_extraction.validate_nexus_python27_path(path)

    assert valid is False
    assert "sieht nicht wie ein Python-Interpreter" in message


def test_validate_nexus_python27_path_accepts_vicon_nexus_python(monkeypatch):
    """
    Test that validate_nexus_python27_path accepts a plausible Nexus Python path.
    """
    path = r"C:\Program Files (x86)\Vicon\Nexus2.12\Python\python.exe"

    monkeypatch.setattr(run_extraction.os.path, "exists", lambda p: True)
    monkeypatch.setattr(run_extraction.os.path, "isfile", lambda p: True)

    valid, message = run_extraction.validate_nexus_python27_path(path)

    assert valid is True
    assert "scheint gültig zu sein" in message


def test_validate_nexus_python27_path_warns_without_vicon_reference(monkeypatch):
    """
    Test that validate_nexus_python27_path returns True with a warning if Vicon is not in the path.
    """
    path = r"C:\Tools\Python\python.exe"

    monkeypatch.setattr(run_extraction.os.path, "exists", lambda p: True)
    monkeypatch.setattr(run_extraction.os.path, "isfile", lambda p: True)

    valid, message = run_extraction.validate_nexus_python27_path(path)

    assert valid is True
    assert "keinen offensichtlichen Vicon-Bezug" in message


def test_resolve_extract_script_path_prefers_existing_candidate(monkeypatch, tmp_path):
    """
    Test that resolve_extract_script_path returns the configured script if it exists.
    """
    script_rel = "src/cmj_framework/vicon_data_retrieval/extraction.py"
    root = tmp_path
    script_path = root / script_rel
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# script", encoding="utf-8")

    monkeypatch.setattr(
        run_extraction,
        "get_vicon_config",
        lambda: {"extract_script": script_rel},
    )
    monkeypatch.setattr(run_extraction, "bundle_root_dir", lambda: root)

    result = run_extraction.resolve_extract_script_path()

    assert result == script_path


def test_resolve_extract_script_path_returns_candidate_even_if_missing(monkeypatch, tmp_path):
    """
    Test that resolve_extract_script_path returns the expected candidate path even if it does not exist.
    """
    script_rel = "src/cmj_framework/vicon_data_retrieval/extraction.py"
    root = tmp_path

    monkeypatch.setattr(
        run_extraction,
        "get_vicon_config",
        lambda: {"extract_script": script_rel},
    )
    monkeypatch.setattr(run_extraction, "bundle_root_dir", lambda: root)
    monkeypatch.setattr(run_extraction, "is_frozen", lambda: False)

    result = run_extraction.resolve_extract_script_path()

    assert result == root / script_rel


def test_run_extraction_returns_none_when_python_is_missing(monkeypatch):
    """
    Test that run_extraction returns None and logs an error if Nexus Python is missing.
    """
    messages = []

    monkeypatch.setattr(run_extraction, "find_nexus_python27", lambda: None)

    result = run_extraction.run_extraction(log_cb=messages.append)

    assert result is None
    assert any("Nexus-Python 2.7 wurde nicht gefunden" in msg for msg in messages)


def test_run_extraction_returns_none_when_script_is_missing(monkeypatch, tmp_path):
    """
    Test that run_extraction returns None if the extraction script does not exist.
    """
    messages = []
    missing_script = tmp_path / "missing_extraction.py"

    monkeypatch.setattr(run_extraction, "find_nexus_python27", lambda: "C:/Vicon/python.exe")
    monkeypatch.setattr(run_extraction, "resolve_extract_script_path", lambda: missing_script)
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: path == "C:/Vicon/python.exe")

    result = run_extraction.run_extraction(log_cb=messages.append)

    assert result is None
    assert any("Extraktionsskript wurde nicht gefunden" in msg for msg in messages)


def test_run_extraction_returns_json_path_on_success(monkeypatch, tmp_path):
    """
    Test that run_extraction returns the extracted JSON path from subprocess output.
    """
    messages = []
    script_path = tmp_path / "extraction.py"
    script_path.write_text("# script", encoding="utf-8")

    class FakeProcess:
        returncode = 0

        def communicate(self):
            return (
                "Hello\nJSON_PATH::C:/data/output.json\nDone\n",
                "",
            )

    monkeypatch.setattr(run_extraction, "find_nexus_python27", lambda: "C:/Vicon/python.exe")
    monkeypatch.setattr(run_extraction, "resolve_extract_script_path", lambda: script_path)
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: str(path) == "C:/Vicon/python.exe")
    monkeypatch.setattr(run_extraction.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = run_extraction.run_extraction(log_cb=messages.append)

    assert result == "C:/data/output.json"
    assert any("JSON_PATH::C:/data/output.json" in msg for msg in messages)


def test_run_extraction_logs_stderr_and_returncode_failure(monkeypatch, tmp_path):
    """
    Test that run_extraction logs stderr lines and a non-zero return code.
    """
    messages = []
    script_path = tmp_path / "extraction.py"
    script_path.write_text("# script", encoding="utf-8")

    class FakeProcess:
        returncode = 1

        def communicate(self):
            return ("", "some error\nanother error\n")

    monkeypatch.setattr(run_extraction, "find_nexus_python27", lambda: "C:/Vicon/python.exe")
    monkeypatch.setattr(run_extraction, "resolve_extract_script_path", lambda: script_path)
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: str(path) == "C:/Vicon/python.exe")
    monkeypatch.setattr(run_extraction.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = run_extraction.run_extraction(log_cb=messages.append)

    assert result is None
    assert any("FEHLER: some error" in msg for msg in messages)
    assert any("FEHLER: another error" in msg for msg in messages)
    assert any("Extraktion fehlgeschlagen" in msg for msg in messages)


def test_run_extraction_returns_none_when_subprocess_raises(monkeypatch, tmp_path):
    """
    Test that run_extraction returns None if subprocess startup fails.
    """
    messages = []
    script_path = tmp_path / "extraction.py"
    script_path.write_text("# script", encoding="utf-8")

    def raise_popen(*args, **kwargs):
        raise OSError("cannot start process")

    monkeypatch.setattr(run_extraction, "find_nexus_python27", lambda: "C:/Vicon/python.exe")
    monkeypatch.setattr(run_extraction, "resolve_extract_script_path", lambda: script_path)
    monkeypatch.setattr(run_extraction.os.path, "exists", lambda path: str(path) == "C:/Vicon/python.exe")
    monkeypatch.setattr(run_extraction.subprocess, "Popen", raise_popen)

    result = run_extraction.run_extraction(log_cb=messages.append)

    assert result is None
    assert any("Extraktion konnte nicht gestartet werden" in msg for msg in messages)