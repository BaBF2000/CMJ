import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cmj_framework.utils.validation import (
    ValidationStatus,
    invalidate_trial_manual,
    log_validation,
    override_validation,
    validate_trial_auto,
)


def test_invalidate_trial_manual_returns_expected_structure():
    """
    Test that manual invalidation returns the expected structure.
    """
    result = invalidate_trial_manual("Bad trial", operator="BBF")

    assert result["status"] == ValidationStatus.INVALID_MANUAL
    assert result["reasons"] == ["Bad trial"]
    assert result["operator"] == "BBF"


def test_override_validation_appends_reason_and_sets_overridden():
    """
    Test that override_validation appends a reason and marks the result as overridden.
    """
    previous = {
        "status": ValidationStatus.INVALID_AUTO,
        "reasons": ["Initial reason"],
    }

    result = override_validation(
        previous_validation=previous,
        new_status=ValidationStatus.VALID,
        reason="Operator approved trial",
    )

    assert result["status"] == ValidationStatus.VALID
    assert result["reasons"] == ["Initial reason", "Operator approved trial"]
    assert result["overridden"] is True


def test_validate_trial_auto_returns_valid_for_clean_trial(monkeypatch):
    """
    Test that a biomechanically plausible trial is marked as valid.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.MIN_JUMP_CM", 5.0)
    monkeypatch.setattr("cmj_framework.utils.validation.MAX_FORCE_BW", 5.0)
    monkeypatch.setattr("cmj_framework.utils.validation.MAX_ASYM_PERCENT", 20.0)
    monkeypatch.setattr("cmj_framework.utils.validation.NOISE_STANDING_THRESH", 0.05)

    roi_T = SimpleNamespace(
        flight_time=0.4,
        jump_height=20.0,
        bodyweight=700.0,
        stand=50,
    )

    metrics = SimpleNamespace(
        takeoff_velocity=2.0,
        asymmetry={
            "Impulse": {
                "Takeoff": {"delta_percent": 10.0},
                "Landing": {"delta_percent": 12.0},
            }
        },
    )

    total_force = np.full(200, 700.0)

    result = validate_trial_auto(
        roi_T=roi_T,
        metrics=metrics,
        total_force=total_force,
        rate=1000.0,
    )

    assert result["status"] == ValidationStatus.VALID
    assert result["reasons"] == []


def test_validate_trial_auto_detects_missing_flight_phase(monkeypatch):
    """
    Test that a missing flight phase is detected.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.MIN_JUMP_CM", 5.0)

    roi_T = SimpleNamespace(
        flight_time=0.0,
        jump_height=20.0,
        bodyweight=700.0,
        stand=50,
    )
    metrics = SimpleNamespace(takeoff_velocity=2.0, asymmetry={})
    total_force = np.full(200, 700.0)

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "No valid flight phase detected" in result["reasons"]


def test_validate_trial_auto_detects_jump_too_low(monkeypatch):
    """
    Test that a jump height below the threshold is detected.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.MIN_JUMP_CM", 10.0)

    roi_T = SimpleNamespace(
        flight_time=0.4,
        jump_height=5.0,
        bodyweight=700.0,
        stand=50,
    )
    metrics = SimpleNamespace(takeoff_velocity=2.0, asymmetry={})
    total_force = np.full(200, 700.0)

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "Jump height < 10.0 cm" in result["reasons"]


def test_validate_trial_auto_detects_non_positive_takeoff_velocity():
    """
    Test that a non-positive take-off velocity is detected.
    """
    roi_T = SimpleNamespace(
        flight_time=0.4,
        jump_height=20.0,
        bodyweight=700.0,
        stand=50,
    )
    metrics = SimpleNamespace(takeoff_velocity=0.0, asymmetry={})
    total_force = np.full(200, 700.0)

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "Non-positive take-off velocity" in result["reasons"]


def test_validate_trial_auto_detects_peak_force_too_high(monkeypatch):
    """
    Test that an excessive peak force relative to bodyweight is detected.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.MAX_FORCE_BW", 2.0)

    roi_T = SimpleNamespace(
        flight_time=0.4,
        jump_height=20.0,
        bodyweight=500.0,
        stand=50,
    )
    metrics = SimpleNamespace(takeoff_velocity=2.0, asymmetry={})
    total_force = np.array([500.0] * 199 + [1200.0])

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "Peak force > 2.0 BW" in result["reasons"]


def test_validate_trial_auto_detects_high_standing_noise(monkeypatch):
    """
    Test that excessive standing-phase noise is detected.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.NOISE_STANDING_THRESH", 0.01)

    roi_T = SimpleNamespace(
        flight_time=0.4,
        jump_height=20.0,
        bodyweight=100.0,
        stand=10,
    )
    metrics = SimpleNamespace(takeoff_velocity=2.0, asymmetry={})
    total_force = np.array([100.0, 120.0, 80.0, 130.0, 70.0, 125.0, 75.0, 110.0, 90.0, 100.0])

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "High noise during standing phase" in result["reasons"]


def test_validate_trial_auto_detects_takeoff_asymmetry(monkeypatch):
    """
    Test that excessive take-off asymmetry is detected.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.MAX_ASYM_PERCENT", 15.0)

    roi_T = SimpleNamespace(
        flight_time=0.4,
        jump_height=20.0,
        bodyweight=700.0,
        stand=50,
    )
    metrics = SimpleNamespace(
        takeoff_velocity=2.0,
        asymmetry={
            "Impulse": {
                "Takeoff": {"delta_percent": 18.0},
                "Landing": {"delta_percent": 10.0},
            }
        },
    )
    total_force = np.full(200, 700.0)

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "Asymmetry > 15.0% during Takeoff" in result["reasons"]


def test_validate_trial_auto_detects_landing_asymmetry(monkeypatch):
    """
    Test that excessive landing asymmetry is detected.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.MAX_ASYM_PERCENT", 15.0)

    roi_T = SimpleNamespace(
        flight_time=0.4,
        jump_height=20.0,
        bodyweight=700.0,
        stand=50,
    )
    metrics = SimpleNamespace(
        takeoff_velocity=2.0,
        asymmetry={
            "Impulse": {
                "Takeoff": {"delta_percent": 10.0},
                "Landing": {"delta_percent": 19.0},
            }
        },
    )
    total_force = np.full(200, 700.0)

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "Asymmetry > 15.0% during Landing" in result["reasons"]


def test_validate_trial_auto_can_return_multiple_reasons(monkeypatch):
    """
    Test that multiple validation issues are accumulated.
    """
    monkeypatch.setattr("cmj_framework.utils.validation.MIN_JUMP_CM", 10.0)

    roi_T = SimpleNamespace(
        flight_time=0.0,
        jump_height=5.0,
        bodyweight=700.0,
        stand=50,
    )
    metrics = SimpleNamespace(takeoff_velocity=0.0, asymmetry={})
    total_force = np.full(200, 700.0)

    result = validate_trial_auto(roi_T, metrics, total_force, rate=1000.0)

    assert result["status"] == ValidationStatus.INVALID_AUTO
    assert "No valid flight phase detected" in result["reasons"]
    assert "Jump height < 10.0 cm" in result["reasons"]
    assert "Non-positive take-off velocity" in result["reasons"]


def test_log_validation_writes_one_jsonl_entry(tmp_path, monkeypatch):
    """
    Test that log_validation appends one JSON line with the expected fields.
    """
    log_file = tmp_path / "validation_log.jsonl"
    monkeypatch.setattr(
        "cmj_framework.utils.validation.get_log_file",
        lambda filename: str(log_file),
    )

    pm = SimpleNamespace(
        patient_name="Max Mustermann",
        session_date="01.01.2020",
    )

    validation_result = {
        "status": ValidationStatus.INVALID_MANUAL,
        "reasons": ["Operator rejected trial"],
        "operator": "BBF",
        "overridden": True,
    }

    output_path = log_validation(
        pm=pm,
        trial="trial_01",
        validation_result=validation_result,
        marker="LASI",
    )

    assert output_path == str(log_file)
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])

    assert entry["patient"] == "Max Mustermann"
    assert entry["session_date"] == "01.01.2020"
    assert entry["trial"] == "trial_01"
    assert entry["marker"] == "LASI"
    assert entry["status"] == ValidationStatus.INVALID_MANUAL
    assert entry["reasons"] == ["Operator rejected trial"]
    assert entry["operator"] == "BBF"
    assert entry["overridden"] is True
    assert "timestamp" in entry


def test_log_validation_writes_none_for_missing_marker(tmp_path, monkeypatch):
    """
    Test that log_validation stores None when no marker is provided.
    """
    log_file = tmp_path / "validation_log.jsonl"
    monkeypatch.setattr(
        "cmj_framework.utils.validation.get_log_file",
        lambda filename: str(log_file),
    )

    pm = SimpleNamespace(
        patient_name="Max Mustermann",
        session_date="01.01.2020",
    )

    validation_result = {
        "status": ValidationStatus.VALID,
        "reasons": [],
    }

    log_validation(
        pm=pm,
        trial="trial_02",
        validation_result=validation_result,
        marker=None,
    )

    entry = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])

    assert entry["marker"] is None
    assert entry["status"] == ValidationStatus.VALID
    assert entry["reasons"] == []
    assert entry["operator"] is None
    assert entry["overridden"] is False