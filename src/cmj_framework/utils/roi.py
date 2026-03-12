from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np

from .signal_processing import SignalProcessing as SP
from .config import G, BW_EST, INIT_VEL, ROUND_VALUE, V_EDGE


class CMKinematics:
    """
    Center of Mass (COM) kinematics for a single vertical force signal.

    Computes position, velocity, and acceleration from the vertical ground
    reaction force using bodyweight-based net force integration.
    """

    def __init__(self, force: np.ndarray, rate: float, bodyweight: Optional[float] = None):
        self.force = np.asarray(force, dtype=float).reshape(-1)
        self.rate = float(rate)

        if self.force.size == 0:
            raise ValueError("Leeres Kraftsignal erkannt.")
        if not np.all(np.isfinite(self.force)):
            raise ValueError("Kraftsignal enthält ungültige Werte.")
        if self.rate <= 0:
            raise ValueError("Ungültige Abtastrate erkannt.")

        # Bodyweight estimation
        if bodyweight is None:
            self.bodyweight, _ = self.estimate_bodyweight()
        else:
            self.bodyweight = float(bodyweight)

        if not np.isfinite(self.bodyweight) or self.bodyweight <= 0:
            raise ValueError("Ungültiges Körpergewicht erkannt.")

        self.mass = self.bodyweight / float(G)
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Ungültige Masse erkannt.")

        self.position, self.velocity, self.acceleration = self.compute_kinematics()

    def estimate_bodyweight(self) -> Tuple[float, int]:
        """
        Estimate bodyweight from the first stable segment of the force signal.

        Returns
        -------
        bw : float
            Estimated bodyweight in Newton.
        idx_end : int
            End index of the detected stable window.
        """
        window_s = float(BW_EST.get("window", 0.5))
        std_thresh = float(BW_EST.get("std_thresh", 0.02))

        n = int(self.rate * window_s)
        if n < 2 or n >= self.force.size:
            raise ValueError("Keine stabile Körpergewicht-Phase erkannt.")

        for i in range(self.force.size - n + 1):
            segment = self.force[i:i + n]
            mean_val = float(np.mean(segment))
            if mean_val == 0:
                continue

            rel_std = float(np.std(segment)) / abs(mean_val)
            if rel_std < std_thresh:
                return float(mean_val), int(i + n)

        raise ValueError("Keine stabile Körpergewicht-Phase erkannt.")

    def compute_kinematics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute acceleration, velocity, and position from the force signal.
        """
        acc = (self.force - self.bodyweight) / self.mass

        vel = np.cumsum(acc) / self.rate
        vel -= self.estimate_initial_velocity(vel)

        pos = np.cumsum(vel) / self.rate
        return pos, vel, acc

    def estimate_initial_velocity(self, vel: np.ndarray) -> float:
        """
        Estimate initial velocity drift from a low-variance velocity window.
        """
        window_s = float(INIT_VEL.get("window", 0.3))
        std_thresh = float(INIT_VEL.get("std_thresh", 0.005))

        n = int(window_s * self.rate)
        n = max(2, min(n, vel.size))

        for i in range(max(1, vel.size - n + 1)):
            segment = vel[i:i + n]
            if float(np.std(segment)) < std_thresh:
                return float(np.mean(segment))

        return float(np.mean(vel[:n]))

    @property
    def bodyweight_N(self) -> float:
        return round(float(self.bodyweight), int(ROUND_VALUE))

    @property
    def mass_kg(self) -> float:
        return round(float(self.mass), int(ROUND_VALUE))


class CMJ_ROI:
    """
    Countermovement Jump (CMJ) ROI detection for a force signal and trajectory.

    Notes
    -----
    - COM kinematics are computed directly from the provided force signal.
    - Takeoff / landing and phase boundaries are then derived from force and
      trajectory features.
    """

    def __init__(self, force: np.ndarray, trajectory: np.ndarray, rate: float):
        self.force = np.asarray(force, dtype=float).reshape(-1)
        self.trajectory = np.asarray(trajectory, dtype=float).reshape(-1)
        self.rate = float(rate)

        if self.force.size == 0 or self.trajectory.size == 0:
            raise ValueError("Leere Signale erkannt.")
        if self.force.size != self.trajectory.size:
            raise ValueError("Force und Trajectory müssen die gleiche Länge haben.")
        if not np.all(np.isfinite(self.force)) or not np.all(np.isfinite(self.trajectory)):
            raise ValueError("Signale enthalten ungültige Werte.")
        if self.rate <= 0:
            raise ValueError("Ungültige Abtastrate erkannt.")

        # Kinematics based on the provided force signal
        self.cm_kin = CMKinematics(self.force, self.rate)
        self.bodyweight = self.cm_kin.bodyweight
        self.mass = self.cm_kin.mass
        self.position = self.cm_kin.position
        self.velocity = self.cm_kin.velocity
        self.acceleration = self.cm_kin.acceleration

        # Flight detection
        self.takeoff_idx, self.landing_idx, self.flight_time = self.detect_takeoff_landing()

        # Detect ROI phases
        self._detect_roi()

    @staticmethod
    def _find_longest_true_run(mask: np.ndarray) -> Tuple[int, int]:
        """
        Return the start/end indices of the longest contiguous True run in a mask.
        """
        best_start = -1
        best_end = -1
        best_len = 0

        current_start = None

        for i, is_true in enumerate(mask):
            if is_true and current_start is None:
                current_start = i
            elif not is_true and current_start is not None:
                current_len = i - current_start
                if current_len > best_len:
                    best_start = current_start
                    best_end = i - 1
                    best_len = current_len
                current_start = None

        if current_start is not None:
            current_len = len(mask) - current_start
            if current_len > best_len:
                best_start = current_start
                best_end = len(mask) - 1
                best_len = current_len

        if best_start < 0 or best_end < 0:
            raise ValueError("Keine zusammenhängende Flugphase erkannt.")

        return int(best_start), int(best_end)

    def detect_takeoff_landing(self) -> Tuple[int, int, float]:
        """
        Detect takeoff and landing indices from force thresholding.

        Returns
        -------
        takeoff_idx : int
        landing_idx : int
        flight_time : float
        """
        std_thresh = float(BW_EST.get("std_thresh", 0.02))
        threshold = float(self.bodyweight) * std_thresh

        below_threshold = self.force < threshold
        if not np.any(below_threshold):
            raise ValueError("Keine Flugphase erkannt (kein Punkt unter Threshold).")

        takeoff_idx, landing_idx = self._find_longest_true_run(below_threshold)

        if landing_idx <= takeoff_idx:
            raise ValueError("Ungültige Flugphase erkannt.")

        flight_time = (landing_idx - takeoff_idx) / self.rate
        return int(takeoff_idx), int(landing_idx), float(flight_time)

    def _detect_roi(self) -> None:
        """
        Detect ROI phases based on force and trajectory features.
        """
        peak = int(np.argmax(self.trajectory))
        if peak <= 1:
            raise ValueError("Ungültiger Trajektorien-Peak erkannt.")

        # Detect the movement onset and local minimum on the trajectory
        self.local_start_t, self.local_min_t = SP.find_v_edge(
            self.trajectory[:peak],
            window_smooth=int(V_EDGE.get("window_smooth", 5)),
            window_slope=int(V_EDGE.get("window_slope", 3)),
        )

        # Detect onset on force up to the trajectory local minimum
        self.local_start_f, self.local_min_f = SP.find_v_edge(
            self.force[: self.local_min_t],
            window_smooth=int(V_EDGE.get("window_smooth", 5)),
            window_slope=int(V_EDGE.get("window_slope", 3)),
        )
        self.start = int(self.local_start_f)

        # Eccentric phases
        self.f_min = int(np.argmin(self.force[: self.local_min_t + 1]))
        self.unloading = (self.start, self.f_min)
        self.braking = (self.f_min, int(self.local_min_t))

        # Standing index = last sample before force minimum at/above bodyweight
        idx_stand = np.where(self.force[: self.f_min + 1] >= self.bodyweight)[0]
        if idx_stand.size == 0:
            raise ValueError("Kein stehender Punkt erkannt.")
        self.stand = int(idx_stand[-1])

        # Jump height based on the provided trajectory
        self._jump_height = round(
            float(np.abs(self.trajectory[self.stand] - self.trajectory[peak])),
            int(ROUND_VALUE),
        )

        # Deceleration phase = first crossing back above BW during braking
        idx_dec = np.where(self.force[self.f_min : self.local_min_t + 1] >= self.bodyweight)[0]
        if idx_dec.size == 0:
            raise ValueError("Kraft erreicht Körpergewicht während Bremsung nicht.")
        self.deceleration = (int(self.f_min + idx_dec[0]), int(self.local_min_t))

        # Concentric propulsion split
        mid = int(self.local_min_t + ((self.takeoff_idx - self.local_min_t) // 2))
        self.propulsion_p1 = (int(self.local_min_t), int(mid))
        self.propulsion_p2 = (int(mid), int(self.takeoff_idx))

        # Define takeoff as the late concentric / terminal push-off segment
        self.takeoff = (int(mid), int(self.takeoff_idx))

        # Landing phase: from landing contact to post-landing trajectory minimum
        post_peak_traj = self.trajectory[peak:]
        min_landing_rel = int(np.argmin(post_peak_traj))
        self.landing = (int(self.landing_idx), int(peak + min_landing_rel))

    @property
    def jump_height(self) -> float:
        return float(self._jump_height)

    @property
    def takeoff_phase(self) -> Tuple[int, int]:
        return self.takeoff

    @property
    def landing_phase(self) -> Tuple[int, int]:
        return self.landing

    @property
    def eccentric_phases(self) -> Dict[str, Tuple[int, int]]:
        return {
            "unloading": self.unloading,
            "braking": self.braking,
            "deceleration": self.deceleration,
        }

    @property
    def concentric_phases(self) -> Dict[str, Tuple[int, int]]:
        return {
            "propulsion_p1": self.propulsion_p1,
            "propulsion_p2": self.propulsion_p2,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "Jump_height": self.jump_height,
            "bodyweight": round(float(self.bodyweight), int(ROUND_VALUE)),
            "mass": round(float(self.mass), int(ROUND_VALUE)),
            "eccentric": self.eccentric_phases,
            "concentric": self.concentric_phases,
            "takeoff": self.takeoff,
            "landing": self.landing,
        }