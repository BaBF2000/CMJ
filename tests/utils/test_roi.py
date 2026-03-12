import numpy as np
import pytest

from cmj_framework.utils.roi import CMKinematics, CMJ_ROI


def test_cmk_raises_for_empty_force():
    """
    Test that CMKinematics raises an error for an empty force signal.
    """
    with pytest.raises(ValueError, match="Leeres Kraftsignal erkannt."):
        CMKinematics(force=np.array([]), rate=1000.0)


def test_cmk_raises_for_non_finite_force():
    """
    Test that CMKinematics raises an error when the force signal contains invalid values.
    """
    force = np.array([700.0, np.nan, 710.0])

    with pytest.raises(ValueError, match="Kraftsignal enthält ungültige Werte."):
        CMKinematics(force=force, rate=1000.0)


def test_cmk_raises_for_invalid_rate():
    """
    Test that CMKinematics raises an error for a non-positive sampling rate.
    """
    force = np.array([700.0, 700.0, 700.0])

    with pytest.raises(ValueError, match="Ungültige Abtastrate erkannt."):
        CMKinematics(force=force, rate=0.0)


def test_cmk_raises_for_invalid_bodyweight():
    """
    Test that CMKinematics raises an error when an invalid bodyweight is provided.
    """
    force = np.full(100, 700.0)

    with pytest.raises(ValueError, match="Ungültiges Körpergewicht erkannt."):
        CMKinematics(force=force, rate=1000.0, bodyweight=0.0)


def test_estimate_bodyweight_returns_expected_value(monkeypatch):
    """
    Test that estimate_bodyweight returns the mean of a stable segment.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.roi.BW_EST",
        {"window": 0.1, "std_thresh": 0.05},
    )

    force = np.full(200, 700.0)
    cmk = CMKinematics(force=force, rate=100.0, bodyweight=700.0)

    bw, idx_end = cmk.estimate_bodyweight()

    assert bw == pytest.approx(700.0)
    assert idx_end == 10


def test_estimate_bodyweight_raises_when_no_stable_window(monkeypatch):
    """
    Test that estimate_bodyweight raises an error when no stable segment is found.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.roi.BW_EST",
        {"window": 0.1, "std_thresh": 0.000001},
    )

    force = np.array([600.0, 800.0] * 20, dtype=float)
    cmk = CMKinematics(force=force, rate=100.0, bodyweight=700.0)

    with pytest.raises(ValueError, match="Keine stabile Körpergewicht-Phase erkannt."):
        cmk.estimate_bodyweight()


def test_estimate_initial_velocity_returns_zero_for_flat_velocity(monkeypatch):
    """
    Test that estimate_initial_velocity returns approximately zero for a flat signal.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.roi.INIT_VEL",
        {"window": 0.1, "std_thresh": 0.01},
    )

    force = np.full(200, 700.0)
    cmk = CMKinematics(force=force, rate=100.0, bodyweight=700.0)

    drift = cmk.estimate_initial_velocity(np.zeros(200))

    assert drift == pytest.approx(0.0)


def test_compute_kinematics_returns_arrays_of_expected_length():
    """
    Test that CMKinematics computes position, velocity, and acceleration arrays
    with the expected shape.
    """
    force = np.full(100, 700.0)
    cmk = CMKinematics(force=force, rate=100.0, bodyweight=700.0)

    assert cmk.position.shape == force.shape
    assert cmk.velocity.shape == force.shape
    assert cmk.acceleration.shape == force.shape


def test_bodyweight_N_property_returns_rounded_value(monkeypatch):
    """
    Test that bodyweight_N returns a rounded bodyweight value.
    """
    monkeypatch.setattr("cmj_framework.utils.roi.ROUND_VALUE", 2)

    force = np.full(100, 700.0)
    cmk = CMKinematics(force=force, rate=100.0, bodyweight=700.1234)

    assert cmk.bodyweight_N == 700.12


def test_mass_kg_property_returns_rounded_value(monkeypatch):
    """
    Test that mass_kg returns a rounded mass value.
    """
    monkeypatch.setattr("cmj_framework.utils.roi.ROUND_VALUE", 3)

    force = np.full(100, 700.0)
    cmk = CMKinematics(force=force, rate=100.0, bodyweight=700.0)

    assert isinstance(cmk.mass_kg, float)


def test_find_longest_true_run_returns_expected_indices():
    """
    Test that the longest contiguous True run is detected correctly.
    """
    mask = np.array([False, True, True, False, True, True, True, False])

    start, end = CMJ_ROI._find_longest_true_run(mask)

    assert start == 4
    assert end == 6


def test_find_longest_true_run_handles_run_at_end():
    """
    Test that _find_longest_true_run handles a True run at the end of the mask.
    """
    mask = np.array([False, True, True, False, True, True, True])

    start, end = CMJ_ROI._find_longest_true_run(mask)

    assert start == 4
    assert end == 6


def test_find_longest_true_run_raises_when_no_true_values():
    """
    Test that _find_longest_true_run raises an error when no True run exists.
    """
    mask = np.array([False, False, False])

    with pytest.raises(ValueError, match="Keine zusammenhängende Flugphase erkannt."):
        CMJ_ROI._find_longest_true_run(mask)


def test_cmj_roi_raises_for_empty_signals():
    """
    Test that CMJ_ROI raises an error for empty signals.
    """
    with pytest.raises(ValueError, match="Leere Signale erkannt."):
        CMJ_ROI(force=np.array([]), trajectory=np.array([]), rate=1000.0)


def test_cmj_roi_raises_for_mismatched_lengths():
    """
    Test that CMJ_ROI raises an error when force and trajectory lengths differ.
    """
    force = np.array([1.0, 2.0, 3.0])
    trajectory = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Force und Trajectory müssen die gleiche Länge haben."):
        CMJ_ROI(force=force, trajectory=trajectory, rate=1000.0)


def test_cmj_roi_raises_for_non_finite_signals():
    """
    Test that CMJ_ROI raises an error when signals contain invalid values.
    """
    force = np.array([700.0, 710.0, np.nan])
    trajectory = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Signale enthalten ungültige Werte."):
        CMJ_ROI(force=force, trajectory=trajectory, rate=1000.0)


def test_cmj_roi_raises_for_invalid_rate():
    """
    Test that CMJ_ROI raises an error for a non-positive sampling rate.
    """
    force = np.array([700.0, 710.0, 720.0])
    trajectory = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Ungültige Abtastrate erkannt."):
        CMJ_ROI(force=force, trajectory=trajectory, rate=0.0)


def test_detect_takeoff_landing_raises_when_no_below_threshold_segment():
    """
    Test that detect_takeoff_landing raises an error when no flight phase is found.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.full(100, 700.0)
    roi.bodyweight = 700.0
    roi.rate = 100.0

    with pytest.raises(ValueError, match="Keine Flugphase erkannt"):
        roi.detect_takeoff_landing()


def test_detect_takeoff_landing_raises_when_no_point_below_threshold(monkeypatch):
    """
    Test that detect_takeoff_landing raises when no force sample falls below threshold.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.array([100.0, 100.0, 100.0], dtype=float)
    roi.bodyweight = 100.0
    roi.rate = 1000.0

    monkeypatch.setattr("cmj_framework.utils.roi.BW_EST", {"std_thresh": 0.02})

    with pytest.raises(ValueError, match="Keine Flugphase erkannt"):
        roi.detect_takeoff_landing()


def test_detect_takeoff_landing_raises_for_invalid_flight_phase(monkeypatch):
    """
    Test that detect_takeoff_landing raises when landing is not after takeoff.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.array([100.0, 1.0, 1.0, 100.0], dtype=float)
    roi.bodyweight = 100.0
    roi.rate = 1000.0

    monkeypatch.setattr(CMJ_ROI, "_find_longest_true_run", lambda self, mask: (3, 3))
    monkeypatch.setattr("cmj_framework.utils.roi.BW_EST", {"std_thresh": 0.02})

    with pytest.raises(ValueError, match="Ungültige Flugphase erkannt"):
        roi.detect_takeoff_landing()


def test_detect_takeoff_landing_returns_expected_indices(monkeypatch):
    """
    Test that detect_takeoff_landing uses the longest below-threshold segment.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.roi.BW_EST",
        {"std_thresh": 0.05},
    )

    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.array(
        [700.0, 700.0, 700.0, 10.0, 5.0, 2.0, 700.0, 700.0],
        dtype=float,
    )
    roi.bodyweight = 700.0
    roi.rate = 100.0

    takeoff_idx, landing_idx, flight_time = roi.detect_takeoff_landing()

    assert takeoff_idx == 3
    assert landing_idx == 5
    assert flight_time == pytest.approx((5 - 3) / 100.0)


def test_jump_height_property_returns_internal_value():
    """
    Test that jump_height returns the stored jump height value.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi._jump_height = 12.34

    assert roi.jump_height == 12.34


def test_takeoff_phase_property_returns_takeoff_tuple():
    """
    Test that takeoff_phase returns the stored takeoff phase.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.takeoff = (10, 20)

    assert roi.takeoff_phase == (10, 20)


def test_landing_phase_property_returns_landing_tuple():
    """
    Test that landing_phase returns the stored landing phase.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.landing = (30, 40)

    assert roi.landing_phase == (30, 40)


def test_eccentric_phases_property_returns_expected_dict():
    """
    Test that eccentric_phases returns the expected dictionary.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.unloading = (1, 2)
    roi.braking = (2, 3)
    roi.deceleration = (3, 4)

    result = roi.eccentric_phases

    assert result == {
        "unloading": (1, 2),
        "braking": (2, 3),
        "deceleration": (3, 4),
    }


def test_concentric_phases_property_returns_expected_dict():
    """
    Test that concentric_phases returns the expected dictionary.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.propulsion_p1 = (4, 5)
    roi.propulsion_p2 = (5, 6)

    result = roi.concentric_phases

    assert result == {
        "propulsion_p1": (4, 5),
        "propulsion_p2": (5, 6),
    }


def test_detect_roi_raises_when_trajectory_peak_is_too_early():
    """
    Test that _detect_roi raises when the trajectory peak is at index <= 1.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.array([100.0, 110.0, 90.0, 105.0], dtype=float)
    roi.trajectory = np.array([0.0, 5.0, 1.0, 0.0], dtype=float)
    roi.bodyweight = 100.0
    roi.takeoff_idx = 3
    roi.landing_idx = 3

    with pytest.raises(ValueError, match="Ungültiger Trajektorien-Peak erkannt"):
        roi._detect_roi()


def test_detect_roi_raises_when_no_standing_point_exists(monkeypatch):
    """
    Test that _detect_roi raises when no standing point is found before force minimum.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.array([80.0, 70.0, 60.0, 110.0, 120.0, 90.0], dtype=float)
    roi.trajectory = np.array([0.0, -1.0, -2.0, -1.0, 5.0, 0.0], dtype=float)
    roi.bodyweight = 100.0
    roi.takeoff_idx = 4
    roi.landing_idx = 5

    calls = [(0, 2), (0, 2)]

    def fake_find_v_edge(*args, **kwargs):
        return calls.pop(0)

    monkeypatch.setattr("cmj_framework.utils.roi.SP.find_v_edge", fake_find_v_edge)
    monkeypatch.setattr("cmj_framework.utils.roi.V_EDGE", {"window_smooth": 5, "window_slope": 3})

    with pytest.raises(ValueError, match="Kein stehender Punkt erkannt"):
        roi._detect_roi()


def test_detect_roi_raises_when_force_never_reaches_bodyweight_during_braking(monkeypatch):
    """
    Test that _detect_roi raises when force never crosses bodyweight during braking.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.array([110.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0], dtype=float)
    roi.trajectory = np.array([0.0, -1.0, -2.0, -1.0, 4.0, 1.0, 0.0], dtype=float)
    roi.bodyweight = 100.0
    roi.takeoff_idx = 5
    roi.landing_idx = 6

    calls = [(0, 3), (0, 3)]

    def fake_find_v_edge(*args, **kwargs):
        return calls.pop(0)

    monkeypatch.setattr("cmj_framework.utils.roi.SP.find_v_edge", fake_find_v_edge)
    monkeypatch.setattr("cmj_framework.utils.roi.V_EDGE", {"window_smooth": 5, "window_slope": 3})

    with pytest.raises(ValueError, match="Kraft erreicht Körpergewicht während Bremsung nicht"):
        roi._detect_roi()


def test_detect_roi_populates_expected_phase_attributes(monkeypatch):
    """
    Test that _detect_roi computes the expected ROI phase attributes.
    """
    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi.force = np.array([110.0, 95.0, 80.0, 120.0, 140.0, 5.0, 5.0, 130.0, 100.0], dtype=float)
    roi.trajectory = np.array([0.0, -1.0, -3.0, -4.0, -1.0, 6.0, 5.0, 3.0, -2.0], dtype=float)
    roi.bodyweight = 100.0
    roi.takeoff_idx = 6
    roi.landing_idx = 7

    calls = [(0, 3), (0, 3)]

    def fake_find_v_edge(*args, **kwargs):
        return calls.pop(0)

    monkeypatch.setattr("cmj_framework.utils.roi.SP.find_v_edge", fake_find_v_edge)
    monkeypatch.setattr("cmj_framework.utils.roi.V_EDGE", {"window_smooth": 5, "window_slope": 3})
    monkeypatch.setattr("cmj_framework.utils.roi.ROUND_VALUE", 2)

    roi._detect_roi()

    assert roi.local_start_t == 0
    assert roi.local_min_t == 3
    assert roi.local_start_f == 0
    assert roi.local_min_f == 3

    assert roi.start == 0
    assert roi.f_min == 2
    assert roi.unloading == (0, 2)
    assert roi.braking == (2, 3)

    assert roi.stand == 0
    assert roi.jump_height == pytest.approx(abs(roi.trajectory[0] - roi.trajectory[5]))

    assert roi.deceleration == (3, 3)
    assert roi.propulsion_p1 == (3, 4)
    assert roi.propulsion_p2 == (4, 6)
    assert roi.takeoff == (4, 6)
    assert roi.landing == (7, 8)


def test_as_dict_returns_expected_structure(monkeypatch):
    """
    Test that as_dict returns the expected dictionary structure.
    """
    monkeypatch.setattr("cmj_framework.utils.roi.ROUND_VALUE", 2)

    roi = CMJ_ROI.__new__(CMJ_ROI)
    roi._jump_height = 15.678
    roi.bodyweight = 700.123
    roi.mass = 71.456
    roi.unloading = (1, 2)
    roi.braking = (2, 3)
    roi.deceleration = (3, 4)
    roi.propulsion_p1 = (4, 5)
    roi.propulsion_p2 = (5, 6)
    roi.takeoff = (6, 7)
    roi.landing = (8, 9)

    result = roi.as_dict()

    assert result["Jump_height"] == 15.678
    assert result["bodyweight"] == 700.12
    assert result["mass"] == 71.46
    assert result["eccentric"] == {
        "unloading": (1, 2),
        "braking": (2, 3),
        "deceleration": (3, 4),
    }
    assert result["concentric"] == {
        "propulsion_p1": (4, 5),
        "propulsion_p2": (5, 6),
    }
    assert result["takeoff"] == (6, 7)
    assert result["landing"] == (8, 9)