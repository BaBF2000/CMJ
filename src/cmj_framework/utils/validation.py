import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .app_logging import get_log_file
from .config import (
    MIN_JUMP_CM,
    MAX_FORCE_BW,
    MAX_ASYM_PERCENT,
    NOISE_STANDING_THRESH,
)
from .pathmanager import PathManager


class ValidationStatus:
    VALID = "VALID"
    INVALID_AUTO = "INVALID_AUTO"
    INVALID_MANUAL = "INVALID_MANUAL"


def validate_trial_auto(
    roi_T: Any,
    metrics: Any,
    total_force: np.ndarray,
    rate: float,
) -> Dict[str, Any]:
    """
    Automatic biomechanical validation of a CMJ trial.
    """
    del rate  # kept for API compatibility

    reasons = []
    T = np.asarray(total_force, dtype=float)

    if getattr(roi_T, "flight_time", 0) <= 0:
        reasons.append("No valid flight phase detected")

    if getattr(roi_T, "jump_height", 0) < MIN_JUMP_CM:
        reasons.append(f"Jump height < {MIN_JUMP_CM} cm")

    if getattr(metrics, "takeoff_velocity", 0) <= 0:
        reasons.append("Non-positive take-off velocity")

    bw = float(getattr(roi_T, "bodyweight", 0.0))
    peak_force = float(np.max(T)) if T.size else 0.0
    if bw > 0 and peak_force > MAX_FORCE_BW * bw:
        reasons.append(f"Peak force > {MAX_FORCE_BW} BW")

    stand_end = int(getattr(roi_T, "stand", 0))
    stand_end = max(0, min(stand_end, len(T)))
    if bw > 0 and stand_end > 1:
        noise = float(np.std(T[:stand_end]) / bw)
        if noise > NOISE_STANDING_THRESH:
            reasons.append("High noise during standing phase")

    # Only Takeoff and Landing asymmetry are validated
    asym = getattr(metrics, "asymmetry", None)
    if isinstance(asym, dict):
        impulse = asym.get("Impulse", {})
        if isinstance(impulse, dict):
            for phase in ("Takeoff", "Landing"):
                vals = impulse.get(phase)
                if isinstance(vals, dict) and vals.get("delta_percent", 0) > MAX_ASYM_PERCENT:
                    reasons.append(f"Asymmetry > {MAX_ASYM_PERCENT}% during {phase}")

    if reasons:
        return {"status": ValidationStatus.INVALID_AUTO, "reasons": reasons}

    return {"status": ValidationStatus.VALID, "reasons": []}


def invalidate_trial_manual(reason: str, operator: Optional[str] = None) -> Dict[str, Any]:
    """Manually invalidate a trial."""
    return {
        "status": ValidationStatus.INVALID_MANUAL,
        "reasons": [reason],
        "operator": operator,
    }


def override_validation(previous_validation: Dict[str, Any], new_status: str, reason: str) -> Dict[str, Any]:
    """Override an existing validation decision."""
    return {
        "status": new_status,
        "reasons": previous_validation.get("reasons", []) + [reason],
        "overridden": True,
    }


def log_validation(
    pm: PathManager,
    trial: Any,
    validation_result: Dict[str, Any],
    marker: Optional[Any] = None,
) -> str:
    """
    Append one validation entry to the central validation log (JSONL).
    """
    path = get_log_file("validation_log.jsonl")

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient": str(pm.patient_name),
        "session_date": str(pm.session_date),
        "trial": str(trial),
        "marker": None if marker is None else str(marker),
        "status": validation_result.get("status", ""),
        "reasons": validation_result.get("reasons", []),
        "operator": validation_result.get("operator"),
        "overridden": validation_result.get("overridden", False),
    }

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return str(path)