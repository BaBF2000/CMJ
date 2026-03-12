from __future__ import annotations

import sys
from pathlib import Path

from cmj_framework.utils.pathmanager import PathManager


def is_frozen() -> bool:
    """Return True if running as a bundled executable."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_root_dir() -> Path:
    """
    Return the runtime root directory.

    Development:
        project root

    PyInstaller:
        directory containing the executable
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def bundle_resource_root() -> Path:
    """
    Return the directory where bundled resources are read from.

    Development:
        project root

    PyInstaller:
        temporary extraction dir (_MEIPASS)
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    """Return the runtime config directory."""
    if is_frozen():
        internal_dir = bundle_root_dir() / "_internal" / "config"
        if internal_dir.exists():
            return internal_dir
        return bundle_root_dir() / "config"
    return bundle_root_dir() / "config"


def config_file(filename: str) -> Path:
    """Return absolute path to a config file."""
    return config_dir() / filename


def documentation_dir() -> Path:
    """Return the documentation directory."""
    if is_frozen():
        internal_dir = bundle_root_dir() / "_internal" / "documentation"
        if internal_dir.exists():
            return internal_dir
        return bundle_root_dir() / "documentation"
    return bundle_root_dir() / "documentation"


def documentation_file(filename: str) -> Path:
    """Return absolute path to a documentation file."""
    return documentation_dir() / filename


def gui_assets_dir() -> Path:
    """Return the GUI assets root directory."""
    if is_frozen():
        internal_dir = bundle_root_dir() / "_internal" / "src" / "cmj_framework" / "gui" / "assets"
        if internal_dir.exists():
            return internal_dir
        return bundle_root_dir() / "src" / "cmj_framework" / "gui" / "assets"
    return bundle_root_dir() / "src" / "cmj_framework" / "gui" / "assets"


def gui_asset(*parts: str) -> Path:
    """Return absolute path to a GUI asset."""
    return gui_assets_dir().joinpath(*parts)


def export_resources_dir() -> Path:
    """Return the bundled export resources directory."""
    if is_frozen():
        internal_dir = bundle_root_dir() / "_internal" / "src" / "cmj_framework"/ "export" / "resources"

        if internal_dir.exists():
            return internal_dir
        return bundle_root_dir() / "src" / "cmj_framework" / "export" / "resources"
    return bundle_root_dir() / "src" / "cmj_framework" / "export" / "resources"


def export_resource_file(filename: str) -> Path:
    """Return absolute path to a bundled export resource file."""
    return export_resources_dir() / filename


def app_data_dir() -> Path:
    """
    Return the main writable CMJ application data directory.

    This is aligned with PathManager canonical storage.
    """
    return Path(PathManager.canonical_base_dir())


def export_user_resources_dir() -> Path:
    """
    Return the writable directory for user-editable export resources.
    """
    return app_data_dir() / "export_resources"


def log_dir() -> Path:
    """
    Return the central writable log directory.
    """
    return app_data_dir() / "Log"


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path