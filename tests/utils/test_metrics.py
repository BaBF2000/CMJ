from types import SimpleNamespace

import numpy as np
import pytest

from cmj_framework.utils.metrics import JumpMetrics


def make_mock_roi():
    """
    Create a minimal ROI-like object for JumpMetrics tests.
    """
    return SimpleNamespace(
        jump_height=0.30,
        bodyweight=100.0,
        mass=10.0,
        start=1,
        stand=1,
        takeoff_phase=(6, 8),
        landing_phase=(9, 11),
        eccentric_phases={
            "braking": (2, 4),
            "deceleration": (3, 4),
        },
        concentric_phases={
            "propulsion_p1": (4, 5),
            "propulsion_p2": (5, 8),
        },
    )


def make_mock_com(velocity):
    """
    Create a minimal COM-like object with a velocity signal.
    """
    return SimpleNamespace(
        velocity=np.asarray(velocity, dtype=float),
    )


def make_metrics_instance():
    """
    Build a JumpMetrics instance with simple deterministic signals and mocked ROI/COM objects.
    """
    L = np.array([10, 10, 20, 30, 40, 50, 60, 70, 20, 30, 40, 20], dtype=float)
    R = np.array([10, 10, 20, 30, 40, 50, 60, 70, 20, 30, 40, 20], dtype=float)
    T = L + R
    X = np.linspace(0.0, 1.0, len(T))
    rate = 10.0

    roi_L = make_mock_roi()
    roi_R = make_mock_roi()
    roi_T = make_mock_roi()

    com = make_mock_com([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.2, 0.1, 0.0])

    return JumpMetrics(
        L=L,
        R=R,
        T=T,
        X=X,
        rate=rate,
        roi_L=roi_L,
        roi_R=roi_R,
        roi_T=roi_T,
        com_L=com,
        com_R=com,
        com_T=com,
    )


def test_init_raises_for_non_positive_sampling_rate():
    """
    Test that JumpMetrics raises an error for a non-positive sampling rate.
    """
    arr = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Sampling rate must be > 0."):
        JumpMetrics(arr, arr, arr, arr, rate=0.0)


def test_init_raises_for_mismatched_signal_lengths():
    """
    Test that JumpMetrics raises an error when input lengths differ.
    """
    L = np.array([1.0, 2.0, 3.0])
    R = np.array([1.0, 2.0])
    T = np.array([2.0, 4.0, 6.0])
    X = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="All input signals must have the same length."):
        JumpMetrics(L, R, T, X, rate=100.0)


def test_init_raises_for_empty_signals():
    """
    Test that JumpMetrics raises an error for empty signals.
    """
    arr = np.array([])

    with pytest.raises(ValueError, match="Input signals must not be empty."):
        JumpMetrics(arr, arr, arr, arr, rate=100.0)


def test_init_raises_for_invalid_values():
    """
    Test that JumpMetrics raises an error when input signals contain invalid values.
    """
    L = np.array([1.0, 2.0, np.nan])
    R = np.array([1.0, 2.0, 3.0])
    T = np.array([2.0, 4.0, 6.0])
    X = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Input signals contain invalid values."):
        JumpMetrics(L, R, T, X, rate=100.0)


def test_slice_inclusive_returns_slice_including_end():
    """
    Test that _slice_inclusive includes the end index.
    """
    s = JumpMetrics._slice_inclusive((2, 4))

    assert s.start == 2
    assert s.stop == 5


def test_slice_inclusive_swaps_reversed_bounds():
    """
    Test that _slice_inclusive handles reversed bounds correctly.
    """
    s = JumpMetrics._slice_inclusive((5, 3))

    assert s.start == 3
    assert s.stop == 6


def test_safe_div_returns_default_for_zero_denominator():
    """
    Test that _safe_div returns the default value when the denominator is zero.
    """
    assert JumpMetrics._safe_div(10.0, 0.0, default=7.0) == 7.0


def test_safe_div_returns_default_for_non_finite_denominator():
    """
    Test that _safe_div returns the default value for non-finite denominators.
    """
    assert JumpMetrics._safe_div(10.0, np.nan, default=3.0) == 3.0


def test_safe_div_returns_expected_ratio():
    """
    Test that _safe_div returns the correct ratio for valid inputs.
    """
    assert JumpMetrics._safe_div(10.0, 2.0) == 5.0


def test_get_phase_window_returns_expected_takeoff_window():
    """
    Test that _get_phase_window returns the expected takeoff phase.
    """
    metrics = make_metrics_instance()

    assert metrics._get_phase_window("takeoff") == (6, 8)


def test_get_phase_window_returns_expected_propulsion_window():
    """
    Test that _get_phase_window returns the combined propulsion window.
    """
    metrics = make_metrics_instance()

    assert metrics._get_phase_window("propulsion") == (4, 8)


def test_get_phase_window_raises_for_unknown_phase():
    """
    Test that _get_phase_window raises an error for an unknown phase.
    """
    metrics = make_metrics_instance()

    with pytest.raises(ValueError, match="Unknown phase: unknown"):
        metrics._get_phase_window("unknown")


def test_peak_force_returns_expected_values():
    """
    Test that peak_force returns the expected peak values for each phase.
    """
    metrics = make_metrics_instance()

    result = metrics.peak_force

    assert result["braking"] == 80.0
    assert result["deceleration"] == 80.0
    assert result["propulsion"] == 140.0
    assert result["landing"] == 80.0


def test_compute_RFD_returns_expected_keys():
    """
    Test that compute_RFD returns the expected keys.
    """
    metrics = make_metrics_instance()

    result = metrics.compute_RFD(windows_ms=[100, 200])

    assert "RFD_max" in result
    assert "RFD_0_100ms" in result
    assert "RFD_0_200ms" in result


def test_compute_RFD_returns_zeroes_for_too_short_phase():
    """
    Test that compute_RFD returns zero values when the braking phase is too short.
    """
    metrics = make_metrics_instance()
    metrics.roi_T.eccentric_phases["braking"] = (2, 2)

    result = metrics.compute_RFD(windows_ms=[100, 200])

    assert result == {
        "RFD_max": 0.0,
        "RFD_0_100ms": 0.0,
        "RFD_0_200ms": 0.0,
    }


def test_RSI_modified_returns_expected_value():
    """
    Test RSI_modified computation.
    """
    metrics = make_metrics_instance()

    result = metrics.RSI_modified

    assert result == pytest.approx(0.429, rel=1e-3)


def test_RSI_modified_returns_zero_when_time_to_takeoff_is_non_positive():
    """
    Test that RSI_modified returns zero when time to takeoff is not positive.
    """
    metrics = make_metrics_instance()
    metrics.roi_T.start = 8
    metrics.roi_T.concentric_phases["propulsion_p2"] = (5, 8)

    assert metrics.RSI_modified == 0.0


def test_takeoff_velocity_returns_expected_value():
    """
    Test takeoff velocity computation from net impulse divided by mass.
    """
    metrics = make_metrics_instance()

    result = metrics.takeoff_velocity

    assert result == pytest.approx(-1.3)


def test_takeoff_velocity_returns_zero_when_interval_is_invalid():
    """
    Test that takeoff_velocity returns zero when the integration interval is invalid.
    """
    metrics = make_metrics_instance()
    metrics.roi_T.stand = 8
    metrics.roi_T.concentric_phases["propulsion_p2"] = (5, 8)

    assert metrics.takeoff_velocity == 0.0


def test_concentric_power_returns_expected_values():
    """
    Test mean and peak concentric power.
    """
    metrics = make_metrics_instance()

    result = metrics.concentric_power

    assert result["Power_mean"] == pytest.approx(44.0)
    assert result["Power_peak"] == pytest.approx(84.0)


def test_concentric_power_returns_zeroes_for_invalid_interval():
    """
    Test that concentric_power returns zeroes when the interval is invalid.
    """
    metrics = make_metrics_instance()
    metrics.roi_T.concentric_phases["propulsion_p1"] = (8, 8)
    metrics.roi_T.concentric_phases["propulsion_p2"] = (5, 8)

    result = metrics.concentric_power

    assert result == {"Power_mean": 0.0, "Power_peak": 0.0}


def test_landing_metrics_returns_expected_values():
    """
    Test landing metrics output structure and values.
    """
    metrics = make_metrics_instance()

    result = metrics.landing_metrics

    assert result["PeakLandingForce"] == 80.0
    assert "LoadingRate_peak" in result


def test_landing_metrics_returns_zeroes_for_invalid_interval():
    """
    Test that landing_metrics returns zeroes when the interval is invalid.
    """
    metrics = make_metrics_instance()
    metrics.roi_T.landing_phase = (5, 5)

    result = metrics.landing_metrics

    assert result == {"LoadingRate_peak": 0.0, "PeakLandingForce": 0.0}


def test_phase_durations_returns_expected_values():
    """
    Test phase duration calculations.
    """
    metrics = make_metrics_instance()

    result = metrics.phase_durations

    assert result == {
        "Eccentric_duration": 0.3,
        "Concentric_duration": 0.4,
        "Time_to_takeoff": 0.7,
    }


def test_asymmetry_returns_zero_delta_for_symmetric_inputs():
    """
    Test that asymmetry delta is zero for perfectly symmetric left/right inputs.
    """
    metrics = make_metrics_instance()

    result = metrics.asymmetry

    for phase_name in result["Impulse"]:
        assert result["Impulse"][phase_name]["delta_percent"] == 0.0

    for phase_name in result["Power"]:
        assert result["Power"][phase_name]["delta_percent"] == 0.0


def test_get_phase_indices_returns_tuple_from_nested_dict_property():
    """
    Test get_phase_indices for a nested dict-like ROI path.
    """
    metrics = make_metrics_instance()

    result = metrics.get_phase_indices(metrics.roi_T, "eccentric_phases.braking")

    assert result == (2, 4)


def test_get_phase_indices_returns_tuple_for_integer_value():
    """
    Test get_phase_indices when the resolved value is a single integer.
    """
    metrics = make_metrics_instance()

    result = metrics.get_phase_indices(metrics.roi_T, "start")

    assert result == (1, 1)


def test_get_phase_indices_raises_for_invalid_value():
    """
    Test get_phase_indices raises an error when the resolved value is invalid.
    """
    metrics = make_metrics_instance()
    metrics.roi_T.invalid_value = "bad"

    with pytest.raises(ValueError, match="Invalid ROI indices for invalid_value"):
        metrics.get_phase_indices(metrics.roi_T, "invalid_value")


def test_all_metrics_returns_expected_top_level_keys():
    """
    Test that all_metrics returns the expected top-level structure.
    """
    metrics = make_metrics_instance()

    result = metrics.all_metrics

    expected_keys = {
        "jump_height",
        "weight",
        "PeakForce",
        "RFD",
        "RSI_modified",
        "Takeoff_velocity",
        "ConcentricPower",
        "LandingMetrics",
        "PhaseDurations",
        "Asymmetry",
    }

    assert set(result.keys()) == expected_keys