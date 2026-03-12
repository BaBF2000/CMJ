import json
from pathlib import Path

import pytest

from cmj_framework.utils.json_manipulation import (
    _ensure_directory,
    create_json,
    delete_json,
    delete_jsons_by_suffix,
    load_json,
    load_jsons_by_suffix,
    update_json,
)


def test_ensure_directory_does_not_raise_for_existing_parent(tmp_path):
    """
    Test that _ensure_directory does not raise when the parent directory exists.
    """
    path = tmp_path / "example.json"

    _ensure_directory(str(path))


def test_ensure_directory_raises_for_missing_parent(tmp_path):
    """
    Test that _ensure_directory raises when the parent directory does not exist.
    """
    path = tmp_path / "missing_dir" / "example.json"

    with pytest.raises(IOError, match="Das Verzeichnis existiert nicht"):
        _ensure_directory(str(path))


def test_load_json_reads_expected_content(tmp_path):
    """
    Test that load_json reads a valid JSON file correctly.
    """
    path = tmp_path / "data.json"
    expected = {"name": "Max", "value": 42}

    path.write_text(json.dumps(expected), encoding="utf-8")

    result = load_json(str(path))

    assert result == expected


def test_load_json_raises_for_missing_file(tmp_path):
    """
    Test that load_json raises when the file does not exist.
    """
    path = tmp_path / "missing.json"

    with pytest.raises(IOError, match="JSON-Datei wurde nicht gefunden"):
        load_json(str(path))


def test_load_json_raises_for_invalid_json(tmp_path):
    """
    Test that load_json raises for malformed JSON content.
    """
    path = tmp_path / "broken.json"
    path.write_text("{invalid json}", encoding="utf-8")

    with pytest.raises(ValueError, match="Fehler beim Parsen der JSON-Datei"):
        load_json(str(path))


def test_load_jsons_by_suffix_reads_matching_files(tmp_path):
    """
    Test that load_jsons_by_suffix loads all matching JSON files.
    """
    file_1 = tmp_path / "trial_01_cmj.json"
    file_2 = tmp_path / "trial_02_cmj.json"
    file_3 = tmp_path / "notes.json"

    file_1.write_text(json.dumps({"id": 1}), encoding="utf-8")
    file_2.write_text(json.dumps({"id": 2}), encoding="utf-8")
    file_3.write_text(json.dumps({"id": 3}), encoding="utf-8")

    result = load_jsons_by_suffix(str(tmp_path), "_cmj.json")

    assert set(result.keys()) == {"trial_01_cmj.json", "trial_02_cmj.json"}
    assert result["trial_01_cmj.json"] == {"id": 1}
    assert result["trial_02_cmj.json"] == {"id": 2}


def test_load_jsons_by_suffix_raises_for_missing_directory(tmp_path):
    """
    Test that load_jsons_by_suffix raises when the directory does not exist.
    """
    missing_dir = tmp_path / "missing"

    with pytest.raises(IOError, match="Ordner wurde nicht gefunden"):
        load_jsons_by_suffix(str(missing_dir), "_cmj.json")


def test_load_jsons_by_suffix_warns_when_no_matching_files(tmp_path):
    """
    Test that load_jsons_by_suffix warns when no matching files are found.
    """
    with pytest.warns(RuntimeWarning, match="Keine JSON-Dateien mit der Endung"):
        result = load_jsons_by_suffix(str(tmp_path), "_cmj.json")

    assert result == {}


def test_load_jsons_by_suffix_warns_and_skips_invalid_file(tmp_path):
    """
    Test that load_jsons_by_suffix warns and skips files that cannot be loaded.
    """
    good_file = tmp_path / "good_cmj.json"
    bad_file = tmp_path / "bad_cmj.json"

    good_file.write_text(json.dumps({"ok": True}), encoding="utf-8")
    bad_file.write_text("{broken json}", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="Fehler beim Laden der Datei"):
        result = load_jsons_by_suffix(str(tmp_path), "_cmj.json")

    assert result == {"good_cmj.json": {"ok": True}}


def test_create_json_writes_file(tmp_path):
    """
    Test that create_json creates a JSON file with the expected content.
    """
    path = tmp_path / "created.json"
    data = {"b": 2, "a": 1}

    create_json(str(path), data)

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data


def test_create_json_raises_when_file_exists_and_no_overwrite(tmp_path):
    """
    Test that create_json raises if the file already exists and overwrite is False.
    """
    path = tmp_path / "created.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    with pytest.raises(IOError, match="Die JSON-Datei existiert bereits"):
        create_json(str(path), {"b": 2}, overwrite=False)


def test_create_json_overwrites_when_allowed(tmp_path):
    """
    Test that create_json overwrites the file when overwrite is True.
    """
    path = tmp_path / "created.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    create_json(str(path), {"b": 2}, overwrite=True)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"b": 2}


def test_update_json_updates_content_and_returns_data(tmp_path):
    """
    Test that update_json modifies the JSON content and returns the updated data.
    """
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"count": 1}), encoding="utf-8")

    def updater(data):
        data["count"] += 1
        data["updated"] = True

    result = update_json(str(path), updater)

    assert result == {"count": 2, "updated": True}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"count": 2, "updated": True}


def test_update_json_raises_when_update_fn_is_not_callable(tmp_path):
    """
    Test that update_json raises when update_fn is not callable.
    """
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"count": 1}), encoding="utf-8")

    with pytest.raises(TypeError, match="update_fn muss eine Funktion sein."):
        update_json(str(path), "not_a_function")


def test_delete_json_removes_existing_file(tmp_path):
    """
    Test that delete_json removes an existing JSON file.
    """
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    delete_json(str(path))

    assert not path.exists()


def test_delete_json_raises_for_missing_file(tmp_path):
    """
    Test that delete_json raises when the file does not exist.
    """
    path = tmp_path / "missing.json"

    with pytest.raises(IOError, match="JSON-Datei konnte nicht gelöscht werden"):
        delete_json(str(path))


def test_delete_jsons_by_suffix_deletes_matching_files(tmp_path):
    """
    Test that delete_jsons_by_suffix deletes all matching files and returns the count.
    """
    file_1 = tmp_path / "trial_01_cmj.json"
    file_2 = tmp_path / "trial_02_cmj.json"
    file_3 = tmp_path / "notes.json"

    file_1.write_text("{}", encoding="utf-8")
    file_2.write_text("{}", encoding="utf-8")
    file_3.write_text("{}", encoding="utf-8")

    deleted_count = delete_jsons_by_suffix(str(tmp_path), "_cmj.json")

    assert deleted_count == 2
    assert not file_1.exists()
    assert not file_2.exists()
    assert file_3.exists()


def test_delete_jsons_by_suffix_raises_for_missing_directory(tmp_path):
    """
    Test that delete_jsons_by_suffix raises when the directory does not exist.
    """
    missing_dir = tmp_path / "missing"

    with pytest.raises(IOError, match="Ordner wurde nicht gefunden"):
        delete_jsons_by_suffix(str(missing_dir), "_cmj.json")


def test_delete_jsons_by_suffix_warns_and_returns_zero_when_no_match(tmp_path):
    """
    Test that delete_jsons_by_suffix warns and returns 0 when no matching files exist.
    """
    with pytest.warns(RuntimeWarning, match="Keine Dateien zum Löschen mit der Endung"):
        result = delete_jsons_by_suffix(str(tmp_path), "_cmj.json")

    assert result == 0