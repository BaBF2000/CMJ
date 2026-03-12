import importlib
import json
import sys
from pathlib import Path

import pytest


MODULE_NAME = "cmj_framework.utils.config"


def reload_config_module(monkeypatch, config_path):
    """
    Reload the config module after patching runtime_paths.config_file.
    """
    import cmj_framework.utils.runtime_paths as runtime_paths

    monkeypatch.setattr(runtime_paths, "config_file", lambda filename: Path(config_path))

    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

    return importlib.import_module(MODULE_NAME)


def test_config_loads_values_from_json(tmp_path, monkeypatch):
    """
    Test that config.py loads values correctly from the JSON file.
    """
    config_data = {
        "round_value": 3,
        "gravity": 9.8,
        "butter_filter": {"cutoff": 12, "order": 2},
        "savgol_filter": {"savgol_window": 31, "savgol_order": 3},
        "bodyweight_estimation": {"window": 0.5, "std_thresh": 0.02},
        "initial_velocity_estimation": {"window": 0.3, "std_thresh": 0.005},
        "takeoff_detection": {"threshold": 10},
        "invalid_trial_criteria": {
            "min_jump_height_cm": 6,
            "max_force_bw": 7,
            "max_asymmetry_percent": 35,
            "noise_standing_thresh": 0.04,
        },
        "units": {"distance": "cm"},
        "v_edge_detection": {"window_smooth": 5, "window_slope": 3},
    }

    config_path = tmp_path / "utils_config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    module = reload_config_module(monkeypatch, config_path)

    assert module.ROUND_VALUE == 3
    assert module.G == 9.8
    assert module.BUTTER_FILTER == {"cutoff": 12, "order": 2}
    assert module.SAVGOL_FILTER == {"savgol_window": 31, "savgol_order": 3}
    assert module.BW_EST == {"window": 0.5, "std_thresh": 0.02}
    assert module.INIT_VEL == {"window": 0.3, "std_thresh": 0.005}
    assert module.TAKEOFF == {"threshold": 10}
    assert module.INVALID == {
        "min_jump_height_cm": 6,
        "max_force_bw": 7,
        "max_asymmetry_percent": 35,
        "noise_standing_thresh": 0.04,
    }
    assert module.UNITS == {"distance": "cm"}
    assert module.V_EDGE == {"window_smooth": 5, "window_slope": 3}

    assert module.MIN_JUMP_CM == 6
    assert module.MAX_FORCE_BW == 7
    assert module.MAX_ASYM_PERCENT == 35
    assert module.NOISE_STANDING_THRESH == 0.04


def test_config_uses_default_values_when_keys_are_missing(tmp_path, monkeypatch):
    """
    Test that config.py falls back to default values when keys are missing.
    """
    config_data = {}

    config_path = tmp_path / "utils_config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    module = reload_config_module(monkeypatch, config_path)

    assert module.ROUND_VALUE == 2
    assert module.G == 9.81

    assert module.BUTTER_FILTER == {}
    assert module.SAVGOL_FILTER == {}
    assert module.BW_EST == {}
    assert module.INIT_VEL == {}
    assert module.TAKEOFF == {}
    assert module.INVALID == {}
    assert module.UNITS == {}
    assert module.V_EDGE == {}

    assert module.MIN_JUMP_CM == 5
    assert module.MAX_FORCE_BW == 8
    assert module.MAX_ASYM_PERCENT == 40
    assert module.NOISE_STANDING_THRESH == 0.05


def test_config_raises_runtime_error_when_file_cannot_be_loaded(tmp_path, monkeypatch):
    """
    Test that config.py raises RuntimeError when the config file cannot be loaded.
    """
    config_path = tmp_path / "utils_config.json"
    config_path.write_text("{invalid json}", encoding="utf-8")

    import cmj_framework.utils.runtime_paths as runtime_paths

    monkeypatch.setattr(runtime_paths, "config_file", lambda filename: Path(config_path))

    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]

    with pytest.raises(RuntimeError, match="Fehler beim Laden der Konfigurationsdatei"):
        importlib.import_module(MODULE_NAME)