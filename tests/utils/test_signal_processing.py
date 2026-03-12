import numpy as np
import pytest

from cmj_framework.utils.signal_processing import SignalProcessing


def test_filter_returns_copy_when_window_is_larger_than_signal(monkeypatch):
    """
    Test that filter returns an unchanged copy when the signal is shorter
    than the configured Savitzky-Golay window.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.signal_processing.SAVGOL_FILTER",
        {"savgol_window": 9, "savgol_order": 3},
    )

    signal = np.array([1.0, 2.0, 3.0, 4.0])
    result = SignalProcessing.filter(signal)

    assert np.array_equal(result, signal)
    assert result is not signal


def test_filter_returns_copy_when_parameters_are_invalid(monkeypatch):
    """
    Test that filter returns an unchanged copy when the Savitzky-Golay
    parameters are invalid.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.signal_processing.SAVGOL_FILTER",
        {"savgol_window": 4, "savgol_order": 3},
    )

    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = SignalProcessing.filter(signal)

    assert np.array_equal(result, signal)
    assert result is not signal


def test_filter_changes_signal_when_parameters_are_valid(monkeypatch):
    """
    Test that filter returns an array with the same shape when parameters
    are valid.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.signal_processing.SAVGOL_FILTER",
        {"savgol_window": 5, "savgol_order": 2},
    )

    signal = np.array([0.0, 1.0, 10.0, 1.0, 0.0, 1.0, 0.0])
    result = SignalProcessing.filter(signal)

    assert result.shape == signal.shape
    assert result.dtype.kind == "f"


def test_butter_lowpass_raises_for_non_positive_sampling_rate():
    """
    Test that butter_lowpass raises an error for a non-positive sampling rate.
    """
    signal = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="Sampling rate must be > 0."):
        SignalProcessing.butter_lowpass(signal, fs=0)


def test_butter_lowpass_raises_for_invalid_cutoff(monkeypatch):
    """
    Test that butter_lowpass raises an error when the cutoff frequency is invalid.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.signal_processing.BUTTER_FILTER",
        {"cutoff": 100, "order": 2},
    )

    signal = np.linspace(0.0, 1.0, 20)

    with pytest.raises(ValueError, match="Ungültige Grenzfrequenz"):
        SignalProcessing.butter_lowpass(signal, fs=100)


def test_butter_lowpass_returns_same_shape(monkeypatch):
    """
    Test that butter_lowpass returns an array with the same shape as the input.
    """
    monkeypatch.setattr(
        "cmj_framework.utils.signal_processing.BUTTER_FILTER",
        {"cutoff": 5, "order": 2},
    )

    t = np.linspace(0.0, 1.0, 200, endpoint=False)
    signal = np.sin(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 20 * t)

    result = SignalProcessing.butter_lowpass(signal, fs=200)

    assert result.shape == signal.shape
    assert result.dtype.kind == "f"


def test_interpolate_raises_when_frame_count_is_invalid():
    """
    Test that interpolate raises an error when frame_count is not greater than 1.
    """
    data = np.array([1.0])

    with pytest.raises(ValueError, match="frame_count and total_samples must be > 1."):
        SignalProcessing.interpolate(frame_count=1, total_samples=10, data=data)


def test_interpolate_raises_when_data_length_does_not_match_frame_count():
    """
    Test that interpolate raises an error when data length does not match frame_count.
    """
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Data length does not match frame_count."):
        SignalProcessing.interpolate(frame_count=4, total_samples=10, data=data)


def test_interpolate_returns_expected_length():
    """
    Test that interpolate returns the requested number of samples.
    """
    data = np.array([0.0, 1.0, 0.0, 1.0])

    result = SignalProcessing.interpolate(frame_count=4, total_samples=20, data=data)

    assert len(result) == 20
    assert result.dtype.kind == "f"


def test_derivatives_returns_empty_list_for_order_less_than_one():
    """
    Test that derivatives returns an empty list when order is less than 1.
    """
    signal = np.array([0.0, 1.0, 2.0, 3.0])

    result = SignalProcessing.derivatives(signal, sampling_rate=100, order=0)

    assert result == []


def test_derivatives_raises_for_non_positive_sampling_rate():
    """
    Test that derivatives raises an error for a non-positive sampling rate.
    """
    signal = np.array([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Sampling rate must be > 0."):
        SignalProcessing.derivatives(signal, sampling_rate=0, order=2)


def test_derivatives_returns_expected_number_of_arrays():
    """
    Test that derivatives returns as many derivative arrays as requested.
    """
    signal = np.linspace(0.0, 10.0, 100)

    result = SignalProcessing.derivatives(signal, sampling_rate=100, order=3)

    assert len(result) == 3
    assert all(isinstance(item, np.ndarray) for item in result)
    assert all(item.shape == signal.shape for item in result)


def test_integrate_raises_for_non_positive_rate():
    """
    Test that integrate raises an error for a non-positive sampling rate.
    """
    signal = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Sampling rate must be > 0."):
        SignalProcessing.integrate(signal, rate=0, start=0, end=2)


def test_integrate_returns_zero_for_empty_signal():
    """
    Test that integrate returns 0.0 for an empty signal.
    """
    signal = np.array([])

    result = SignalProcessing.integrate(signal, rate=100, start=0, end=1)

    assert result == 0.0


def test_integrate_computes_expected_trapezoidal_value():
    """
    Test that integrate computes the expected trapezoidal integral.
    """
    signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result = SignalProcessing.integrate(signal, rate=2, start=0, end=4)

    assert result == pytest.approx(2.0)


def test_integrate_swaps_start_and_end_when_needed():
    """
    Test that integrate handles reversed index bounds correctly.
    """
    signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result = SignalProcessing.integrate(signal, rate=2, start=4, end=0)

    assert result == pytest.approx(2.0)


def test_convert_distance_converts_mm_to_cm():
    """
    Test conversion from millimeters to centimeters.
    """
    result = SignalProcessing.convert_distance(10.0, from_unit="mm", to_unit="cm")

    assert result == pytest.approx(1.0)


def test_convert_distance_converts_m_to_mm():
    """
    Test conversion from meters to millimeters.
    """
    result = SignalProcessing.convert_distance(1.5, from_unit="m", to_unit="mm")

    assert result == pytest.approx(1500.0)


def test_convert_distance_raises_for_invalid_unit():
    """
    Test that convert_distance raises an error for invalid units.
    """
    with pytest.raises(ValueError, match="Einheit muss 'mm', 'cm' oder 'm' sein."):
        SignalProcessing.convert_distance(10.0, from_unit="km", to_unit="m")


def test_find_v_edge_raises_for_empty_array():
    """
    Test that find_v_edge raises an error for an empty array.
    """
    with pytest.raises(ValueError, match="Leeres Array"):
        SignalProcessing.find_v_edge(np.array([]))


def test_find_v_edge_raises_for_invalid_direction():
    """
    Test that find_v_edge raises an error for an invalid direction.
    """
    signal = np.array([5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError, match="Richtung muss 'left' oder 'right' sein."):
        SignalProcessing.find_v_edge(signal, direction="up")


def test_find_v_edge_returns_valid_indices_for_left_and_right():
    """
    Test that find_v_edge returns coherent indices for both directions.
    """
    signal = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    left_edge, left_min = SignalProcessing.find_v_edge(signal, direction="left")
    right_edge, right_min = SignalProcessing.find_v_edge(signal, direction="right")

    assert left_min == right_min
    assert left_edge <= left_min <= right_edge
    assert 0 <= left_edge < len(signal)
    assert 0 <= right_edge < len(signal)