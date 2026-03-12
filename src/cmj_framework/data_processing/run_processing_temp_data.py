import os
from typing import Optional, Dict, Callable

import numpy as np

from cmj_framework.utils.json_manipulation import load_json
from cmj_framework.utils.signal_processing import SignalProcessing


LogCB = Callable[[str], None]


class TempProcessedData:
    """
    Multi-trial singleton to store and access CMJ data in memory.
    """
    _trials: Dict[str, "TempProcessedData"] = {}

    def __new__(cls, trial_name: str = "current"):
        if trial_name not in cls._trials:
            instance = super().__new__(cls)
            cls._trials[trial_name] = instance
            instance._initialized = False
        return cls._trials[trial_name]

    def __init__(self, trial_name: str = "current"):
        if getattr(self, "_initialized", False):
            return

        self.trial_name = trial_name
        self.Fz_l: Optional[np.ndarray] = None
        self.Fz_r: Optional[np.ndarray] = None
        self.F_total: Optional[np.ndarray] = None
        self.trajectory: Optional[np.ndarray] = None
        self.frame_rate: Optional[float] = None
        self.plate_rate: Optional[float] = None
        self.json_path: Optional[str] = None
        self.validation_result = None

        self._initialized = True

    @classmethod
    def remove_trial(cls, trial_name: str) -> bool:
        """Remove a trial from memory."""
        if trial_name in cls._trials:
            del cls._trials[trial_name]
            return True
        return False

    @classmethod
    def load(
        cls,
        json_path: str,
        trial_name: str = "current",
        log_cb: Optional[LogCB] = None,
        force_reload: bool = False,
    ) -> "TempProcessedData":
        """
        Create or get an instance, load JSON if needed, and return the object.

        If the trial is already loaded from the same path, reuse it directly.
        """
        abs_json_path = os.path.abspath(json_path)
        instance = cls(trial_name)

        already_loaded = (
            not force_reload
            and instance.json_path is not None
            and os.path.abspath(instance.json_path) == abs_json_path
            and instance.F_total is not None
            and instance.trajectory is not None
        )

        if already_loaded:
            if log_cb:
                log_cb(f"[{trial_name}] Bereits im Speicher, erneutes Laden übersprungen.")
            return instance

        instance.process_cmj_data(abs_json_path, log_cb=log_cb)
        return instance

    @classmethod
    def get_trial(cls, trial_name: str) -> Optional["TempProcessedData"]:
        """Return an existing trial instance."""
        return cls._trials.get(trial_name)

    @classmethod
    def list_trials(cls):
        """List all loaded trials."""
        return list(cls._trials.keys())

    def process_cmj_data(
        self,
        json_path: str,
        log_cb: Optional[LogCB] = None,
    ) -> None:
        """Read the JSON and store values in memory."""
        self.json_path = os.path.abspath(json_path)

        data = load_json(json_path)
        cache = data.get("data_cache", {})

        def _log(message: str) -> None:
            if log_cb:
                log_cb(message)

        # ---------------- FORCES ----------------
        Fz_l_raw = np.asarray(cache["forces"]["L"]["z"]["raw"], dtype=float)
        Fz_r_raw = np.asarray(cache["forces"]["R"]["z"]["raw"], dtype=float)

        self.Fz_l = SignalProcessing.filter(Fz_l_raw)
        self.Fz_r = SignalProcessing.filter(Fz_r_raw)
        self.F_total = SignalProcessing.filter(Fz_l_raw + Fz_r_raw)

        # ---------------- VALID MARKER ----------------
        valid_marker = cache.get("valid_markers")
        if not valid_marker:
            raise ValueError(f"[{self.trial_name}] Kein gültiger Marker im JSON definiert")

        markers = cache.get("markers", {})
        if valid_marker not in markers:
            raise ValueError(f"[{self.trial_name}] Gültiger Marker '{valid_marker}' nicht gefunden")

        marker_data = markers[valid_marker]
        if "z" not in marker_data or marker_data["z"] is None:
            raise ValueError(
                f"[{self.trial_name}] Marker '{valid_marker}' hat keine verwertbare Z-Trajektorie"
            )

        traj_raw = np.asarray(marker_data["z"], dtype=float)

        # ---------------- TRAJECTORY PROCESSING ----------------
        self.trajectory = SignalProcessing.butter_lowpass(
            SignalProcessing.convert_distance(traj_raw),
            data.get("user_info", {}).get("framerate", 200.0),
        )

        # ---------------- TRIAL INFO ----------------
        self.frame_rate = float(data.get("user_info", {}).get("framerate", 200.0))
        self.plate_rate = float(data.get("user_info", {}).get("platerate", 1000.0))

        # Reset validation when data is reloaded
        self.validation_result = None

        # ---------------- INTERPOLATION ----------------
        n_force = len(self.F_total)
        n_traj = len(self.trajectory)

        if n_force != n_traj:
            _log(
                "[{}] Unterschiedliche Längen: force={}, traj={}. Interpolation.".format(
                    self.trial_name,
                    n_force,
                    n_traj,
                )
            )
            self.trajectory = SignalProcessing.interpolate(n_traj, n_force, self.trajectory)

    def get_data(self) -> dict:
        """Return data as a dictionary."""
        return {
            "Fz_l": self.Fz_l,
            "Fz_r": self.Fz_r,
            "F_total": self.F_total,
            "trajectory": self.trajectory,
            "frame_rate": self.frame_rate,
            "plate_rate": self.plate_rate,
            "validation_result": self.validation_result,
        }
    