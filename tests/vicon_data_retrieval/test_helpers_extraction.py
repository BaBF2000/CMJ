import json
from pathlib import Path

import numpy as np
import pytest

from cmj_framework.vicon_data_retrieval.helpers_extraction import (
    DataCache,
    JSONExporter,
    MarkerQualityLogger,
    UserDataExtractor,
)


class FakeViconForUserData:
    def __init__(self, trial_path, trial_name, framecount=100, framerate=100, platerate=1000):
        self._trial_path = trial_path
        self._trial_name = trial_name
        self._framecount = framecount
        self._framerate = framerate
        self._platerate = platerate

    def GetTrialName(self):
        return self._trial_path, self._trial_name

    def GetFrameCount(self):
        return self._framecount

    def GetFrameRate(self):
        return self._framerate

    def GetDeviceDetails(self, _device_id):
        return ("ForcePlate1", "ForcePlate", self._platerate)


class FakeViconForDataCache:
    def __init__(self, subjects):
        self._subjects = subjects

    def GetSubjectNames(self):
        return self._subjects


class FakeViconForExporter:
    def __init__(self, trial_path, trial_name):
        self._trial_path = trial_path
        self._trial_name = trial_name

    def GetTrialName(self):
        return self._trial_path, self._trial_name


class FakeUserData:
    def __init__(self, data):
        self._data = data

    def as_dict(self):
        return dict(self._data)


class FakeDataCache:
    def __init__(self, data):
        self._data = data

    def load_data(self):
        return self._data


class FakePathManager:
    def __init__(self, raw_dir):
        self._raw_dir = Path(raw_dir)

    def raw_file(self, filename):
        return str(self._raw_dir / filename)


def test_user_data_extractor_extracts_name_correctly():
    """
    Test that UserDataExtractor formats the patient name from the folder structure.
    """
    trial_path = str(Path("C:/data/Max Mustermann/01.01.2020/trials"))
    trial_name = "1234_01012020_001"

    vicon = FakeViconForUserData(trial_path=trial_path, trial_name=trial_name)
    extractor = UserDataExtractor(vicon)

    assert extractor.name == "Max, Mustermann"


def test_user_data_extractor_returns_unknown_patient_when_name_cannot_be_extracted():
    """
    Test that UserDataExtractor falls back to 'Unknown Patient' when extraction fails.
    """
    trial_path = ""
    trial_name = "1234_01012020_001"

    vicon = FakeViconForUserData(trial_path=trial_path, trial_name=trial_name)
    extractor = UserDataExtractor(vicon)

    assert extractor.name == "Unknown Patient"


def test_user_data_extractor_extracts_trial_date_correctly():
    """
    Test that UserDataExtractor parses the trial date from the trial name.
    """
    trial_path = str(Path("C:/data/Max Mustermann/01.01.2020/trials"))
    trial_name = "1234_01012020_001"

    vicon = FakeViconForUserData(trial_path=trial_path, trial_name=trial_name)
    extractor = UserDataExtractor(vicon)

    assert extractor.trial_date == "01.01.2020"


def test_user_data_extractor_returns_none_for_invalid_trial_date():
    """
    Test that UserDataExtractor returns None when the trial date cannot be parsed.
    """
    trial_path = str(Path("C:/data/Max Mustermann/01.01.2020/trials"))
    trial_name = "invalid_trial_name"

    vicon = FakeViconForUserData(trial_path=trial_path, trial_name=trial_name)
    extractor = UserDataExtractor(vicon)

    assert extractor.trial_date is None


def test_user_data_extractor_as_dict_returns_expected_structure():
    """
    Test that UserDataExtractor.as_dict returns all expected fields.
    """
    trial_path = str(Path("C:/data/Max Mustermann/01.01.2020/trials"))
    trial_name = "1234_01012020_001"

    vicon = FakeViconForUserData(
        trial_path=trial_path,
        trial_name=trial_name,
        framecount=150,
        framerate=120,
        platerate=1080,
    )
    extractor = UserDataExtractor(vicon)

    result = extractor.as_dict()

    assert result == {
        "name": "Max, Mustermann",
        "trial_date": "01.01.2020",
        "framecount": 150,
        "framerate": 120,
        "platerate": 1080,
    }


def test_marker_quality_logger_tracks_valid_and_invalid_markers():
    """
    Test that MarkerQualityLogger stores valid markers and invalid reasons correctly.
    """
    logger = MarkerQualityLogger()

    logger.mark_valid("LASI")
    logger.mark_invalid("RASI", "missing trajectory")
    logger.mark_invalid("RASI", "contains NaN")
    logger.mark_invalid("RASI", "missing trajectory")

    assert logger.valid == {"LASI"}
    assert logger.invalid == {"RASI": ["missing trajectory", "contains NaN"]}


def test_marker_quality_logger_has_invalid_returns_expected_value():
    """
    Test that has_invalid reflects whether invalid markers were recorded.
    """
    logger = MarkerQualityLogger()
    assert logger.has_invalid() is False

    logger.mark_invalid("LASI", "missing")
    assert logger.has_invalid() is True


def test_marker_quality_logger_summary_returns_sorted_valid_markers():
    """
    Test that summary returns the expected structure and sorted valid markers.
    """
    logger = MarkerQualityLogger()
    logger.mark_valid("RASI")
    logger.mark_valid("LASI")
    logger.mark_invalid("LPSI", "missing")

    result = logger.summary()

    assert result == {
        "valid_markers": ["LASI", "RASI"],
        "invalid_markers": {"LPSI": ["missing"]},
    }


def test_data_cache_get_current_subject_returns_first_subject():
    """
    Test that DataCache.get_current_subject returns the first available subject.
    """
    vicon = FakeViconForDataCache(subjects=["SubjectA", "SubjectB"])
    cache = DataCache(vicon=vicon, config={}, axes=["z"])

    assert cache.subject == "SubjectA"


def test_data_cache_get_current_subject_raises_when_no_subjects_exist():
    """
    Test that DataCache raises when no subjects are available.
    """
    vicon = FakeViconForDataCache(subjects=[])

    with pytest.raises(RuntimeError, match="Keine Patienten in den Vicon-Daten gefunden."):
        DataCache(vicon=vicon, config={}, axes=["z"])


def test_json_exporter_to_json_safe_converts_numpy_types():
    """
    Test that to_json_safe converts numpy arrays and scalar types to JSON-safe values.
    """
    exporter = JSONExporter(vicon=None)

    data = {
        "array": np.array([1, 2, 3]),
        "float": np.float64(1.5),
        "int": np.int64(4),
        "nested": {
            "list": [np.float32(2.5), np.int32(7)],
        },
    }

    result = exporter.to_json_safe(data)

    assert result == {
        "array": [1, 2, 3],
        "float": 1.5,
        "int": 4,
        "nested": {
            "list": [2.5, 7],
        },
    }


def test_json_exporter_save_writes_json_to_pathmanager_raw_dir(tmp_path):
    """
    Test that JSONExporter.save writes to PathManager raw_data when new_path is True.
    """
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir(parents=True)

    vicon = FakeViconForExporter(
        trial_path=str(tmp_path / "trial_folder"),
        trial_name='1234_01012020_001',
    )
    exporter = JSONExporter(vicon=vicon)

    user_data = FakeUserData(
        {
            "name": "Max, Mustermann",
            "trial_date": "01.01.2020",
            "framecount": 100,
            "framerate": 100,
            "platerate": 1000,
        }
    )
    data_cache = FakeDataCache(
        {
            "forces": {"Kistler": {"z": {"raw": [1, 2, 3]}}},
            "markers": {"LASI": {"z": [4, 5, 6]}},
            "marker_quality": {"valid_markers": ["LASI"], "invalid_markers": {}},
            "valid_markers": "LASI",
        }
    )
    path_manager = FakePathManager(raw_dir=raw_dir)

    json_path = exporter.save(
        data_cache=data_cache,
        user_data=user_data,
        path_manager=path_manager,
        new_path=True,
    )

    expected_path = raw_dir / "1234_01012020_001_data_cmj.json"

    assert Path(json_path) == expected_path
    assert expected_path.exists()

    saved = json.loads(expected_path.read_text(encoding="utf-8"))
    assert saved["user_info"]["name"] == "Max, Mustermann"
    assert saved["data_cache"]["valid_markers"] == "LASI"


def test_json_exporter_save_writes_json_next_to_trial_when_new_path_is_false(tmp_path):
    """
    Test that JSONExporter.save writes next to the trial path when new_path is False.
    """
    trial_dir = tmp_path / "trial_dir"
    trial_dir.mkdir()

    vicon = FakeViconForExporter(
        trial_path=str(trial_dir / "trial_file"),
        trial_name="1234_01012020_001",
    )
    exporter = JSONExporter(vicon=vicon)

    user_data = FakeUserData({"name": "Max, Mustermann"})
    data_cache = FakeDataCache({"forces": {}, "markers": {}, "marker_quality": {}, "valid_markers": None})
    path_manager = FakePathManager(raw_dir=tmp_path / "unused_raw")

    json_path = exporter.save(
        data_cache=data_cache,
        user_data=user_data,
        path_manager=path_manager,
        new_path=False,
    )

    expected_path = trial_dir / "1234_01012020_001_data_cmj.json"

    assert Path(json_path) == expected_path
    assert expected_path.exists()


def test_json_exporter_save_uses_custom_filename(tmp_path):
    """
    Test that JSONExporter.save uses the provided custom filename.
    """
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir(parents=True)

    vicon = FakeViconForExporter(
        trial_path=str(tmp_path / "trial_dir" / "trial_file"),
        trial_name="1234_01012020_001",
    )
    exporter = JSONExporter(vicon=vicon)

    user_data = FakeUserData({"name": "Max, Mustermann"})
    data_cache = FakeDataCache({"forces": {}, "markers": {}, "marker_quality": {}, "valid_markers": None})
    path_manager = FakePathManager(raw_dir=raw_dir)

    json_path = exporter.save(
        data_cache=data_cache,
        user_data=user_data,
        path_manager=path_manager,
        new_path=True,
        filename="custom_name.json",
    )

    expected_path = raw_dir / "custom_name.json"

    assert Path(json_path) == expected_path
    assert expected_path.exists()


def test_json_exporter_save_sanitizes_trial_name_for_filename(tmp_path):
    """
    Test that JSONExporter.save sanitizes forbidden characters in the trial name.
    """
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir(parents=True)

    vicon = FakeViconForExporter(
        trial_path=str(tmp_path / "trial_dir" / "trial_file"),
        trial_name='bad:/\\*?"<>|name',
    )
    exporter = JSONExporter(vicon=vicon)

    user_data = FakeUserData({"name": "Max, Mustermann"})
    data_cache = FakeDataCache({"forces": {}, "markers": {}, "marker_quality": {}, "valid_markers": None})
    path_manager = FakePathManager(raw_dir=raw_dir)

    json_path = exporter.save(
        data_cache=data_cache,
        user_data=user_data,
        path_manager=path_manager,
        new_path=True,
    )

    assert Path(json_path).name == "bad_________name_data_cmj.json"