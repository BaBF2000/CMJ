from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List

import numpy as np

from .signal_processing import SignalProcessing as SP
from .config import ROUND_VALUE
from .roi import CMJ_ROI


class JumpMetrics:
    """
    Compute performance metrics for a single CMJ trial.

    Requires
    --------
    - CMJ_ROI objects for left, right, and total forces
      (or raw signals to compute them)
    - COM kinematics (available through roi.cm_kin by default)
    """

    def __init__(
        self,
        L: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        rate: float,
        roi_L: Optional[CMJ_ROI] = None,
        roi_R: Optional[CMJ_ROI] = None,
        roi_T: Optional[CMJ_ROI] = None,
        com_L: Any = None,
        com_R: Any = None,
        com_T: Any = None,
    ):
        self.L = np.asarray(L, dtype=float).reshape(-1)
        self.R = np.asarray(R, dtype=float).reshape(-1)
        self.T = np.asarray(T, dtype=float).reshape(-1)
        self.X = np.asarray(X, dtype=float).reshape(-1)
        self.rate = float(rate)

        if self.rate <= 0:
            raise ValueError("Sampling rate must be > 0.")

        sizes = {self.L.size, self.R.size, self.T.size, self.X.size}
        if len(sizes) != 1:
            raise ValueError("All input signals must have the same length.")
        if self.L.size == 0:
            raise ValueError("Input signals must not be empty.")
        if not all(np.all(np.isfinite(arr)) for arr in (self.L, self.R, self.T, self.X)):
            raise ValueError("Input signals contain invalid values.")

        # ROI computation
        self.roi_L = roi_L or CMJ_ROI(self.L, self.X, self.rate)
        self.roi_R = roi_R or CMJ_ROI(self.R, self.X, self.rate)
        self.roi_T = roi_T or CMJ_ROI(self.T, self.X, self.rate)

        # COM kinematics
        self.com_L = com_L or self.roi_L.cm_kin
        self.com_R = com_R or self.roi_R.cm_kin
        self.com_T = com_T or self.roi_T.cm_kin

    # ---------------- Internal helpers ----------------
    @staticmethod
    def _slice_inclusive(span: Tuple[int, int]) -> slice:
        """Return a Python slice that includes the end index."""
        s, e = int(span[0]), int(span[1])
        if e < s:
            s, e = e, s
        return slice(s, e + 1)

    @staticmethod
    def _safe_div(num: float, den: float, default: float = 0.0) -> float:
        """Safe division helper."""
        if den == 0 or not np.isfinite(den):
            return default
        return num / den

    def _get_phase_window(self, phase: str) -> Tuple[int, int]:
        """
        Return a common total-signal phase window for asymmetry calculations.

        A single shared window is used for L, R, and T to ensure direct
        left-right comparability within the same time interval.
        """
        r = self.roi_T

        if phase == "takeoff":
            return r.takeoff_phase

        if phase == "braking":
            return r.eccentric_phases["braking"]

        if phase == "deceleration":
            return r.eccentric_phases["deceleration"]

        if phase == "propulsion":
            return (
                int(r.concentric_phases["propulsion_p1"][0]),
                int(r.concentric_phases["propulsion_p2"][1]),
            )

        if phase == "landing":
            return r.landing_phase

        raise ValueError(f"Unknown phase: {phase}")

    # ---------------- Peak Forces ----------------
    @property
    def peak_force(self) -> Dict[str, float]:
        r = self.roi_T
        return {
            "braking": round(float(np.max(self.T[self._slice_inclusive(r.eccentric_phases["braking"])])), ROUND_VALUE),
            "deceleration": round(float(np.max(self.T[self._slice_inclusive(r.eccentric_phases["deceleration"])])), ROUND_VALUE),
            "propulsion": round(
                float(
                    np.max(
                        self.T[
                            self._slice_inclusive(
                                (r.concentric_phases["propulsion_p1"][0], r.concentric_phases["propulsion_p2"][1])
                            )
                        ]
                    )
                ),
                ROUND_VALUE,
            ),
            "landing": round(float(np.max(self.T[self._slice_inclusive(r.landing_phase)])), ROUND_VALUE),
        }

    # ---------------- Rate of Force Development ----------------
    def compute_RFD(self, windows_ms: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Compute RFD metrics during the braking phase.

        Parameters
        ----------
        windows_ms : list of int, optional
            Time windows in milliseconds, e.g. [100, 200].
        """
        if windows_ms is None:
            windows_ms = [100, 200]

        df = SP.derivatives(self.T, self.rate, 1)[0]
        s, e = self.roi_T.eccentric_phases["braking"]
        s = int(s)
        e = int(e)

        if e <= s + 1:
            return {"RFD_max": 0.0, **{f"RFD_0_{w}ms": 0.0 for w in windows_ms}}

        out: Dict[str, float] = {"RFD_max": round(float(np.max(df[s:e + 1])), ROUND_VALUE)}

        for w in windows_ms:
            n = int(self.rate * float(w) / 1000.0)
            if n <= 0 or (e - s) <= n:
                out[f"RFD_0_{w}ms"] = 0.0
                continue

            best = None
            for i in range(s, e - n + 1):
                slope = (self.T[i + n] - self.T[i]) / (n / self.rate)
                best = slope if best is None else max(best, slope)

            out[f"RFD_0_{w}ms"] = round(float(best if best is not None else 0.0), ROUND_VALUE)

        return out

    # ---------------- RSI Modified ----------------
    @property
    def RSI_modified(self) -> float:
        r = self.roi_T
        time_to_takeoff = (r.concentric_phases["propulsion_p2"][1] - r.start) / self.rate
        if time_to_takeoff <= 0:
            return 0.0
        return round(float(r.jump_height) / float(time_to_takeoff), 3)

    # ---------------- Takeoff Velocity ----------------
    @property
    def takeoff_velocity(self) -> float:
        r = self.roi_T
        s = int(r.stand)
        e = int(r.concentric_phases["propulsion_p2"][1])

        if e <= s:
            return 0.0

        net_force = self.T[s:e + 1] - float(r.bodyweight)
        impulse = float(np.trapz(net_force, dx=1.0 / self.rate))

        return round(float(self._safe_div(impulse, float(r.mass), default=0.0)), ROUND_VALUE)

    # ---------------- Concentric Power ----------------
    @property
    def concentric_power(self) -> Dict[str, float]:
        r = self.roi_T
        s = int(r.concentric_phases["propulsion_p1"][0])
        e = int(r.concentric_phases["propulsion_p2"][1])

        if e <= s:
            return {"Power_mean": 0.0, "Power_peak": 0.0}

        power = self.T[s:e + 1] * np.asarray(self.com_T.velocity[s:e + 1], dtype=float)

        return {
            "Power_mean": round(float(np.mean(power)), ROUND_VALUE),
            "Power_peak": round(float(np.max(power)), ROUND_VALUE),
        }

    # ---------------- Landing Metrics ----------------
    @property
    def landing_metrics(self) -> Dict[str, float]:
        s, e = self.roi_T.landing_phase
        s = int(s)
        e = int(e)

        if e <= s:
            return {"LoadingRate_peak": 0.0, "PeakLandingForce": 0.0}

        df = SP.derivatives(self.T, self.rate, 1)[0]
        return {
            "LoadingRate_peak": round(float(np.max(df[s:e + 1])), 1),
            "PeakLandingForce": round(float(np.max(self.T[s:e + 1])), 1),
        }

    # ---------------- Phase Durations ----------------
    @property
    def phase_durations(self) -> Dict[str, float]:
        r = self.roi_T
        return {
            "Eccentric_duration": round((r.eccentric_phases["braking"][1] - r.start) / self.rate, 3),
            "Concentric_duration": round(
                (r.concentric_phases["propulsion_p2"][1] - r.concentric_phases["propulsion_p1"][0]) / self.rate, 3
            ),
            "Time_to_takeoff": round((r.concentric_phases["propulsion_p2"][1] - r.start) / self.rate, 3),
        }

    # ---------------- Left-Right Asymmetry ----------------
    @property
    def asymmetry(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute left-right asymmetry for impulse and power over key jump phases.

        Asymmetry index formula
        -----------------------
            ASI = |L - R| / ((|L| + |R|) / 2) * 100

        Contributions are expressed relative to the total value in the same phase.
        """
        phases = ["takeoff", "braking", "deceleration", "propulsion", "landing"]
        out: Dict[str, Dict[str, Dict[str, float]]] = {"Impulse": {}, "Power": {}}

        V_T = np.asarray(self.com_T.velocity, dtype=float)

        for phase in phases:
            s, e = self._get_phase_window(phase)
            s = int(s)
            e = int(e)

            if e < s:
                s, e = e, s

            # ---------------- Impulse ----------------
            iL = float(SP.integrate(self.L, self.rate, s, e))
            iR = float(SP.integrate(self.R, self.rate, s, e))
            iT = float(SP.integrate(self.T, self.rate, s, e))

            abs_iL = abs(iL)
            abs_iR = abs(iR)
            mean_impulse = (abs_iL + abs_iR) / 2.0

            delta_impulse = self._safe_div(abs(abs_iL - abs_iR) * 100.0, mean_impulse, default=0.0)

            if abs(iT) == 0:
                contrib_L = 50.0
                contrib_R = 50.0
            else:
                contrib_L = abs_iL / abs(iT) * 100.0
                contrib_R = abs_iR / abs(iT) * 100.0

            out["Impulse"][phase.capitalize()] = {
                "L": round(iL, ROUND_VALUE),
                "R": round(iR, ROUND_VALUE),
                "delta_percent": round(float(delta_impulse), ROUND_VALUE),
                "contribution_L_percent": round(float(contrib_L), ROUND_VALUE),
                "contribution_R_percent": round(float(contrib_R), ROUND_VALUE),
            }

            # ---------------- Power ----------------
            F_L = self.L[s:e + 1]
            F_R = self.R[s:e + 1]
            F_T = self.T[s:e + 1]
            V_phase_T = V_T[s:e + 1]

            pL = float(np.mean(F_L * V_phase_T)) if F_L.size else 0.0
            pR = float(np.mean(F_R * V_phase_T)) if F_R.size else 0.0
            pT = float(np.mean(F_T * V_phase_T)) if F_T.size else 0.0

            abs_pL = abs(pL)
            abs_pR = abs(pR)
            mean_power = (abs_pL + abs_pR) / 2.0

            delta_power = self._safe_div(abs(abs_pL - abs_pR) * 100.0, mean_power, default=0.0)

            if abs(pT) == 0:
                contrib_PL = 50.0
                contrib_PR = 50.0
            else:
                contrib_PL = abs_pL / abs(pT) * 100.0
                contrib_PR = abs_pR / abs(pT) * 100.0

            out["Power"][phase.capitalize()] = {
                "L": round(pL, ROUND_VALUE),
                "R": round(pR, ROUND_VALUE),
                "delta_percent": round(float(delta_power), ROUND_VALUE),
                "contribution_L_percent": round(float(contrib_PL), ROUND_VALUE),
                "contribution_R_percent": round(float(contrib_PR), ROUND_VALUE),
            }

        return out

    def get_phase_indices(self, roi: Any, attr_path: str) -> Tuple[int, int]:
        """
        Retrieve start and end indices from an ROI attribute string.

        Supports nested attributes and dicts, e.g.:
            "eccentric_phases.braking"
        """
        attrs = attr_path.split(".")
        val: Any = roi

        for attr in attrs:
            val = getattr(val, attr) if hasattr(val, attr) else val[attr]

        if isinstance(val, tuple) and len(val) == 2:
            return int(val[0]), int(val[1])

        if isinstance(val, int):
            return int(val), int(val)

        raise ValueError(f"Invalid ROI indices for {attr_path}")

    # ---------------- Full Metrics ----------------
    @property
    def all_metrics(self) -> Dict[str, Any]:
        return {
            "jump_height": self.roi_T.jump_height,
            "weight": self.roi_T.bodyweight,
            "PeakForce": self.peak_force,
            "RFD": self.compute_RFD(),
            "RSI_modified": self.RSI_modified,
            "Takeoff_velocity": self.takeoff_velocity,
            "ConcentricPower": self.concentric_power,
            "LandingMetrics": self.landing_metrics,
            "PhaseDurations": self.phase_durations,
            "Asymmetry": self.asymmetry,
        }