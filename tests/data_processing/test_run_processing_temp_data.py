import numpy as np
import pytest

from cmj_framework.data_processing.run_processing_temp_data import TempProcessedData


@pytest.fixture(autouse=True)
def clear_temp_processed_data_trials():
    """
    Ensure the singleton cache is reset before and after each test.
    """
    TempProcessedData._trials.clear()
    yield
    TempProcessedData._trials.clear()


def make_valid_json_payload(
    force_left=None,
    force_right=None,
    marker_z=None,
    valid_marker="LASI",
    framerate=200.0,
    platerate=1000.0,
):
    """
    Create a minimal valid JSON-like payload for TempProcessedData tests.
    """
    if force_left is None:
        force_left = [100.0, 110.0, 120.0, 130.0]
    if force_right is None:
        force_right = [90.0, 100.0, 110.0, 120.0]
    if marker_z is None:
        marker_z = [1000.0, 1010.0, 1020.0, 1030.0]

    return {
        "user_info": {
            "framerate": framerate,
            "platerate": platerate,
        },
        "data_cache": {
            "forces": {
                "L": {"z": {"raw": force_left}},
                "R": {"z": {"raw": force_right}},
            },
            "valid_markers": valid_marker,
            "markers": {
                valid_marker: {
                    "z": marker_z,
                }
            },
        },
    }


def test_same_trial_name_returns_same_instance():
    """
    Test that the singleton returns the same instance for the same trial name.
    """
    a = TempProcessedData("trial_1")
    b = TempProcessedData("trial_1")

    assert a is b


def test_different_trial_names_return_different_instances():
    """
    Test that different trial names create different instances.
    """
    a = TempProcessedData("trial_1")
    b = TempProcessedData("trial_2")

    assert a is not b


def test_remove_trial_returns_true_when_trial_exists():
    """
    Test that remove_trial returns True when a cached trial is removed.
    """
    TempProcessedData("trial_1")

    result = TempProcessedData.remove_trial("trial_1")

    assert result is True
    assert TempProcessedData.get_trial("trial_1") is None


def test_remove_trial_returns_false_when_trial_does_not_exist():
    """
    Test that remove_trial returns False when the trial does not exist.
    """
    result = TempProcessedData.remove_trial("missing_trial")

    assert result is False


def test_get_trial_returns_existing_instance():
    """
    Test that get_trial returns the cached instance.
    """
    instance = TempProcessedData("trial_1")

    result = TempProcessedData.get_trial("trial_1")

    assert result is instance


def test_list_trials_returns_loaded_trial_names():
    """
    Test that list_trials returns all loaded trial names.
    """
    TempProcessedData("trial_a")
    TempProcessedData("trial_b")

    result = TempProcessedData.list_trials()

    assert set(result) == {"trial_a", "trial_b"}


def test_process_cmj_data_loads_and_stores_expected_values(monkeypatch):
    """
    Test that process_cmj_data loads force and trajectory data correctly.
    """
    payload = make_valid_json_payload()

    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.load_json",
        lambda path: payload,
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.filter",
        lambda x: np.asarray(x, dtype=float),
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.convert_distance",
        lambda x: np.asarray(x, dtype=float) / 10.0,
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.butter_lowpass",
        lambda data, fs: np.asarray(data, dtype=float),
    )

    temp = TempProcessedData("trial_1")
    temp.process_cmj_data("fake_path.json")

    assert np.array_equal(temp.Fz_l, np.array([100.0, 110.0, 120.0, 130.0]))
    assert np.array_equal(temp.Fz_r, np.array([90.0, 100.0, 110.0, 120.0]))
    assert np.array_equal(temp.F_total, np.array([190.0, 210.0, 230.0, 250.0]))
    assert np.array_equal(temp.trajectory, np.array([100.0, 101.0, 102.0, 103.0]))
    assert temp.frame_rate == 200.0
    assert temp.plate_rate == 1000.0
    assert temp.validation_result is None
    assert temp.json_path.endswith("fake_path.json")


def test_process_cmj_data_raises_when_valid_marker_is_missing(monkeypatch):
    """
    Test that process_cmj_data raises when no valid marker is defined.
    """
    payload = make_valid_json_payload()
    payload["data_cache"]["valid_markers"] = None

    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.load_json",
        lambda path: payload,
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.filter",
        lambda x: np.asarray(x, dtype=float),
    )

    temp = TempProcessedData("trial_1")

    with pytest.raises(ValueError, match="Kein gültiger Marker im JSON definiert"):
        temp.process_cmj_data("fake_path.json")


def test_process_cmj_data_raises_when_valid_marker_not_found(monkeypatch):
    """
    Test that process_cmj_data raises when the declared valid marker is missing from markers.
    """
    payload = make_valid_json_payload(valid_marker="LASI")
    payload["data_cache"]["markers"] = {"RASI": {"z": [1, 2, 3, 4]}}

    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.load_json",
        lambda path: payload,
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.filter",
        lambda x: np.asarray(x, dtype=float),
    )

    temp = TempProcessedData("trial_1")

    with pytest.raises(ValueError, match="Gültiger Marker 'LASI' nicht gefunden"):
        temp.process_cmj_data("fake_path.json")


def test_process_cmj_data_raises_when_marker_has_no_z_trajectory(monkeypatch):
    """
    Test that process_cmj_data raises when the valid marker has no usable z trajectory.
    """
    payload = make_valid_json_payload()
    payload["data_cache"]["markers"]["LASI"] = {"x": [1, 2, 3, 4]}

    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.load_json",
        lambda path: payload,
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.filter",
        lambda x: np.asarray(x, dtype=float),
    )

    temp = TempProcessedData("trial_1")

    with pytest.raises(ValueError, match="hat keine verwertbare Z-Trajektorie"):
        temp.process_cmj_data("fake_path.json")


def test_process_cmj_data_interpolates_when_lengths_differ(monkeypatch):
    """
    Test that process_cmj_data interpolates the trajectory when lengths differ.
    """
    payload = make_valid_json_payload(
        force_left=[100.0, 110.0, 120.0, 130.0],
        force_right=[90.0, 100.0, 110.0, 120.0],
        marker_z=[1000.0, 1010.0],
    )

    log_messages = []

    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.load_json",
        lambda path: payload,
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.filter",
        lambda x: np.asarray(x, dtype=float),
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.convert_distance",
        lambda x: np.asarray(x, dtype=float),
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.butter_lowpass",
        lambda data, fs: np.asarray(data, dtype=float),
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.interpolate",
        lambda frame_count, total_samples, data: np.linspace(data[0], data[-1], total_samples),
    )

    temp = TempProcessedData("trial_1")
    temp.process_cmj_data("fake_path.json", log_cb=log_messages.append)

    assert len(temp.trajectory) == len(temp.F_total)
    assert any("Interpolation" in message for message in log_messages)


def test_load_reuses_existing_loaded_trial_without_reloading(monkeypatch):
    """
    Test that load reuses an already loaded trial from the same path.
    """
    instance = TempProcessedData("trial_1")
    instance.json_path = "/abs/path/file.json"
    instance.F_total = np.array([1.0, 2.0])
    instance.trajectory = np.array([3.0, 4.0])

    called = {"count": 0}

    def fake_process(self, json_path, log_cb=None):
        called["count"] += 1

    monkeypatch.setattr(TempProcessedData, "process_cmj_data", fake_process)

    messages = []
    result = TempProcessedData.load(
        "/abs/path/file.json",
        trial_name="trial_1",
        log_cb=messages.append,
        force_reload=False,
    )

    assert result is instance
    assert called["count"] == 0
    assert any("Bereits im Speicher" in message for message in messages)


def test_load_forces_reload_when_force_reload_is_true(monkeypatch):
    """
    Test that load forces a reload when force_reload is True.
    """
    instance = TempProcessedData("trial_1")
    instance.json_path = "/abs/path/file.json"
    instance.F_total = np.array([1.0, 2.0])
    instance.trajectory = np.array([3.0, 4.0])

    called = {"count": 0}

    def fake_process(self, json_path, log_cb=None):
        called["count"] += 1

    monkeypatch.setattr(TempProcessedData, "process_cmj_data", fake_process)

    result = TempProcessedData.load(
        "/abs/path/file.json",
        trial_name="trial_1",
        force_reload=True,
    )

    assert result is instance
    assert called["count"] == 1


def test_load_calls_process_when_trial_not_already_loaded(monkeypatch):
    """
    Test that load processes the JSON when the trial is not already loaded.
    """
    captured = {"path": None}

    def fake_process(self, json_path, log_cb=None):
        captured["path"] = json_path
        self.json_path = json_path
        self.F_total = np.array([1.0])
        self.trajectory = np.array([1.0])

    monkeypatch.setattr(TempProcessedData, "process_cmj_data", fake_process)

    result = TempProcessedData.load("relative_file.json", trial_name="trial_1")

    assert isinstance(result, TempProcessedData)
    assert captured["path"].endswith("relative_file.json")


def test_get_data_returns_expected_dictionary(monkeypatch):
    """
    Test that get_data returns all stored data fields.
    """
    payload = make_valid_json_payload()

    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.load_json",
        lambda path: payload,
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.filter",
        lambda x: np.asarray(x, dtype=float),
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.convert_distance",
        lambda x: np.asarray(x, dtype=float),
    )
    monkeypatch.setattr(
        "cmj_framework.data_processing.run_processing_temp_data.SignalProcessing.butter_lowpass",
        lambda data, fs: np.asarray(data, dtype=float),
    )

    temp = TempProcessedData("trial_1")
    temp.process_cmj_data("fake_path.json")
    temp.validation_result = {"status": "VALID"}

    result = temp.get_data()

    assert set(result.keys()) == {
        "Fz_l",
        "Fz_r",
        "F_total",
        "trajectory",
        "frame_rate",
        "plate_rate",
        "validation_result",
    }
    assert result["validation_result"] == {"status": "VALID"}