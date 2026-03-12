import numpy as np

from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

from .config import BUTTER_FILTER, SAVGOL_FILTER


class SignalProcessing:
    """Signal processing utilities for CMJ analysis."""

    @staticmethod
    def filter(signal: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing if parameters are valid.
        If invalid, return a copy of the input signal.
        """
        x = np.asarray(signal, dtype=float)

        window = int(SAVGOL_FILTER.get("savgol_window", 31))
        order = int(SAVGOL_FILTER.get("savgol_order", 3))

        # SciPy constraints:
        # - window_length must be odd and >= 3
        # - polyorder must be < window_length
        if window < 3 or window % 2 == 0 or order < 0 or order >= window:
            return x.copy()

        if x.size < window:
            return x.copy()

        return savgol_filter(x, window_length=window, polyorder=order)

    @staticmethod
    def butter_lowpass(data: np.ndarray, fs: float) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter.
        """
        x = np.asarray(data, dtype=float)
        fs = float(fs)
        if fs <= 0:
            raise ValueError("Sampling rate must be > 0.")

        cutoff = float(BUTTER_FILTER.get("cutoff", 12))
        order = int(BUTTER_FILTER.get("order", 2))

        nyq = 0.5 * fs
        if cutoff <= 0 or cutoff >= nyq:
            # Keep user-facing messages consistent with the project style
            raise ValueError("Ungültige Grenzfrequenz für Butterworth-Filter.")

        b, a = butter(order, cutoff / nyq, btype="low", analog=False)
        return filtfilt(b, a, x)

    @staticmethod
    def interpolate(frame_count: int, total_samples: int, data: np.ndarray) -> np.ndarray:
        """
        Resample a signal using cubic interpolation.
        frame_count: original number of samples
        total_samples: target number of samples
        """
        frame_count = int(frame_count)
        total_samples = int(total_samples)
        y = np.asarray(data, dtype=float)

        if frame_count <= 1 or total_samples <= 1:
            raise ValueError("frame_count and total_samples must be > 1.")
        if y.size != frame_count:
            raise ValueError("Data length does not match frame_count.")

        f = interp1d(np.linspace(0, 1, frame_count), y, kind="cubic")
        return f(np.linspace(0, 1, total_samples))

    @staticmethod
    def derivatives(signal: np.ndarray, sampling_rate: float, order: int = 2) -> list[np.ndarray]:
        """
        Compute successive numerical derivatives using numpy.gradient.
        Returns a list of derivatives [d1, d2, ...].
        """
        x = np.asarray(signal, dtype=float)
        sampling_rate = float(sampling_rate)
        order = int(order)

        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be > 0.")
        if order < 1:
            return []

        dt = 1.0 / sampling_rate
        derivatives_list: list[np.ndarray] = []
        current = x

        for _ in range(order):
            current = np.gradient(current, dt)
            derivatives_list.append(current)

        return derivatives_list

    @staticmethod
    def integrate(signal: np.ndarray, rate: float, start: int, end: int) -> float:
        """
        Compute the integral (impulse) using the trapezoidal rule.
        Indices are inclusive: [start, end].
        """
        x = np.asarray(signal, dtype=float)
        rate = float(rate)
        start = int(start)
        end = int(end)
    
        if rate <= 0:
            raise ValueError("Sampling rate must be > 0.")
        if x.size == 0:
            return 0.0
    
        start = max(0, min(start, x.size - 1))
        end = max(0, min(end, x.size - 1))
        if end < start:
            start, end = end, start
    
        return float(np.trapz(x[start:end + 1], dx=1.0 / rate))

    @staticmethod
    def convert_distance(value: np.ndarray, from_unit: str = "mm", to_unit: str = "cm") -> float:
        """
        Convert distances between mm, cm, and m.
        """
        units = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
        if from_unit not in units or to_unit not in units:
            raise ValueError("Einheit muss 'mm', 'cm' oder 'm' sein.")
        return value * units[from_unit] / units[to_unit]

    @staticmethod
    def find_v_edge(
        array: np.ndarray,
        direction: str = "left",
        window_smooth: int = 5,
        window_slope: int = 3,
    ) -> tuple[int, int]:
        """
        Find the left or right boundary of a V-shaped segment in a 1D signal.

        Returns
        -------
        idx_edge : int
            Detected boundary index.
        idx_min : int
            Index of the local minimum (on the smoothed signal).
        """
        x = np.asarray(array, dtype=float)
        if x.size == 0:
            raise ValueError("Leeres Array für V-Edge-Erkennung.")

        window_smooth = max(1, int(window_smooth))
        window_slope = max(1, int(window_slope))

        x_smooth = uniform_filter1d(x, size=window_smooth)
        idx_min = int(np.argmin(x_smooth))
        idx = idx_min

        if direction == "left":
            step = -1
            stop = 0
            slope_sign = 1
        elif direction == "right":
            step = 1
            stop = len(x_smooth) - 1
            slope_sign = -1
        else:
            raise ValueError("Richtung muss 'left' oder 'right' sein.")

        while idx != stop:
            next_idx = idx + step

            if direction == "left":
                start_idx = max(0, idx - window_slope)
            else:
                start_idx = min(len(x_smooth) - 1, idx + window_slope)

            slope = x_smooth[next_idx] - x_smooth[start_idx]
            if slope * slope_sign >= 0:
                break

            idx = next_idx

        return int(idx), int(idx_min)

