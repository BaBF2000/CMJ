"""
Utility package for the CMJ framework.

Public API:
- configuration constants
- JSON helpers
- signal processing helpers
- ROI / kinematics
- metrics computation
- path manager
- plotting utilities
- runtime path helpers
"""

from .config import (
    config,
    CONFIG_PATH,
    ROUND_VALUE,
    G,
    BUTTER_FILTER,
    SAVGOL_FILTER,
    BW_EST,
    INIT_VEL,
    TAKEOFF,
    INVALID,
    UNITS,
    V_EDGE,
    MIN_JUMP_CM,
    MAX_FORCE_BW,
    MAX_ASYM_PERCENT,
    NOISE_STANDING_THRESH,
)

from .json_manipulation import (
    load_json,
    load_jsons_by_suffix,
    create_json,
    update_json,
    delete_json,
    delete_jsons_by_suffix,
)

from .signal_processing import SignalProcessing
from .roi import CMKinematics, CMJ_ROI
from .metrics import JumpMetrics
from .pathmanager import PathManager

# Plotting is optional (matplotlib dependency)
try:
    from .visualisation import CMJPlot
except ImportError:
    CMJPlot = None

from .runtime_paths import (
    is_frozen,
    bundle_root_dir,
    bundle_resource_root,
    config_dir,
    config_file,
    documentation_dir,
    documentation_file,
    gui_assets_dir,
    gui_asset,
    export_resources_dir,
    export_resource_file,
    app_data_dir,
    export_user_resources_dir,
    log_dir,
    ensure_dir,
)

__all__ = [
    # config
    "config",
    "CONFIG_PATH",
    "ROUND_VALUE",
    "G",
    "BUTTER_FILTER",
    "SAVGOL_FILTER",
    "BW_EST",
    "INIT_VEL",
    "TAKEOFF",
    "INVALID",
    "UNITS",
    "V_EDGE",
    "MIN_JUMP_CM",
    "MAX_FORCE_BW",
    "MAX_ASYM_PERCENT",
    "NOISE_STANDING_THRESH",

    # json
    "load_json",
    "load_jsons_by_suffix",
    "create_json",
    "update_json",
    "delete_json",
    "delete_jsons_by_suffix",

    # core utils
    "SignalProcessing",
    "CMKinematics",
    "CMJ_ROI",
    "JumpMetrics",
    "PathManager",
    "CMJPlot",

    # runtime_paths
    "is_frozen",
    "bundle_root_dir",
    "bundle_resource_root",
    "config_dir",
    "config_file",
    "documentation_dir",
    "documentation_file",
    "gui_assets_dir",
    "gui_asset",
    "export_resources_dir",
    "export_resource_file",
    "app_data_dir",
    "export_user_resources_dir",
    "log_dir",
    "ensure_dir",
]