import json

from cmj_framework.utils.runtime_paths import config_file


CONFIG_PATH = config_file("utils_config.json")

try:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
except Exception as exc:
    raise RuntimeError(
        f"Fehler beim Laden der Konfigurationsdatei:\n{CONFIG_PATH}\n{exc}"
    ) from exc


# ------------------------------------------------------------
# Global numeric settings
# ------------------------------------------------------------

ROUND_VALUE = config.get("round_value", 2)
G = config.get("gravity", 9.81)


# ------------------------------------------------------------
# Sub-config blocks (with fallback defaults)
# ------------------------------------------------------------

BUTTER_FILTER = config.get("butter_filter", {})
SAVGOL_FILTER = config.get("savgol_filter", {})
BW_EST = config.get("bodyweight_estimation", {})
INIT_VEL = config.get("initial_velocity_estimation", {})
TAKEOFF = config.get("takeoff_detection", {})
INVALID = config.get("invalid_trial_criteria", {})
UNITS = config.get("units", {})
V_EDGE = config.get("v_edge_detection", {})


# ------------------------------------------------------------
# Validation thresholds (derived constants)
# ------------------------------------------------------------

MIN_JUMP_CM = INVALID.get("min_jump_height_cm", 5)
MAX_FORCE_BW = INVALID.get("max_force_bw", 8)
MAX_ASYM_PERCENT = INVALID.get("max_asymmetry_percent", 40)
NOISE_STANDING_THRESH = INVALID.get("noise_standing_thresh", 0.05)