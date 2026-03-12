import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from cmj_framework.utils.runtime_paths import (
    bundle_resource_root,
    config_file,
    is_frozen,
)


DEFAULT_NEXUS_PY27 = r"C:\Program Files (x86)\Vicon\Nexus2.12\Python\python.exe"
DEFAULT_EXTRACT_SCRIPT = "src/cmj_framework/vicon_data_retrieval/extraction.py"


def bundle_root_dir() -> Path:
    """
    Resolve the root directory for packaged resources/scripts.

    Development:
        project root

    PyInstaller:
        temporary extraction dir (_MEIPASS)
    """
    return bundle_resource_root()


def app_config_path() -> Path:
    """Return active Vicon path config file."""
    return config_file("vicon_path_config.json")


def default_app_config_path() -> Path:
    """Return default Vicon path config file."""
    return config_file("vicon_path_config.default.json")


def ensure_app_config_exists() -> Path:
    """
    Ensure the active Vicon path config exists.

    If missing:
    - create it from the default config if available
    - otherwise create it from fallback content
    """
    cfg_path = app_config_path()
    if cfg_path.exists():
        return cfg_path

    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    default_path = default_app_config_path()
    if default_path.exists():
        cfg_path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        fallback = {
            "vicon": {
                "nexus_python27_path": r"C:/Program Files (x86)/Vicon/Nexus2.12/Python/python.exe",
                "extract_script": DEFAULT_EXTRACT_SCRIPT,
                "nexus_search_roots": [
                    r"C:/Program Files (x86)/Vicon",
                    r"C:/Program Files/Vicon",
                ],
            }
        }
        cfg_path.write_text(
            json.dumps(fallback, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return cfg_path


def load_app_config() -> dict:
    """Load the Vicon path config JSON."""
    cfg_path = ensure_app_config_exists()
    try:
        with open(cfg_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def get_vicon_config() -> dict:
    """Return the 'vicon' section from the config."""
    cfg = load_app_config()
    if not isinstance(cfg, dict):
        return {}

    vicon_cfg = cfg.get("vicon", {})
    return vicon_cfg if isinstance(vicon_cfg, dict) else {}


def parse_nexus_version_from_path(path_str: str) -> tuple[int, int, int]:
    """
    Extract version numbers from a path containing '.../NexusX.Y[.Z]/...'.

    Unknown -> (0, 0, 0).
    """
    parts = Path(path_str).parts
    nexus_part = None

    for part in parts:
        if part.lower().startswith("nexus"):
            nexus_part = part
            break

    if not nexus_part:
        return (0, 0, 0)

    version_text = nexus_part[5:]
    numbers: list[int] = []

    for token in version_text.split("."):
        try:
            numbers.append(int(token))
        except Exception:
            numbers.append(0)

    while len(numbers) < 3:
        numbers.append(0)

    return tuple(numbers[:3])


def find_nexus_python27() -> Optional[str]:
    """
    Try to locate the Vicon Nexus Python 2.7 interpreter.

    Search order:
    1) Explicit path from config/vicon_path_config.json
    2) Environment override: CMJ_NEXUS_PY27
    3) Automatic scan from configured search roots
    4) Fallback common roots
    5) DEFAULT_NEXUS_PY27
    """
    vicon_cfg = get_vicon_config()

    configured_path = str(vicon_cfg.get("nexus_python27_path", "")).strip().strip('"')
    if configured_path and os.path.exists(configured_path):
        return configured_path

    env_path = os.environ.get("CMJ_NEXUS_PY27", "").strip().strip('"')
    if env_path and os.path.exists(env_path):
        return env_path

    config_roots = vicon_cfg.get("nexus_search_roots", [])
    roots = [str(root) for root in config_roots if str(root).strip()]

    if not roots:
        roots = [
            r"C:\Program Files (x86)\Vicon",
            r"C:\Program Files\Vicon",
        ]

    candidates: list[str] = []

    for root in roots:
        candidates.extend(glob.glob(os.path.join(root, "Nexus*", "Python", "python*.exe")))

    for root in roots:
        candidates.extend(glob.glob(os.path.join(root, "Nexus*", "python*.exe")))

    existing = [candidate for candidate in candidates if os.path.exists(candidate)]

    if existing:
        existing.sort(key=parse_nexus_version_from_path, reverse=True)
        return existing[0]

    if os.path.exists(DEFAULT_NEXUS_PY27):
        return DEFAULT_NEXUS_PY27

    return None


def validate_nexus_python27_path(path: str) -> tuple[bool, str]:
    """
    Validate a candidate Nexus Python interpreter path.

    Returns
    -------
    (is_valid, message)
    """
    candidate = str(path or "").strip().strip('"')

    if not candidate:
        return False, "Kein Pfad eingetragen."

    if not os.path.exists(candidate):
        return False, f"Datei nicht gefunden:\n{candidate}"

    if not os.path.isfile(candidate):
        return False, f"Pfad ist keine Datei:\n{candidate}"

    name = os.path.basename(candidate).lower()
    if name not in ("python.exe", "pythonw.exe"):
        return False, (
            "Die Datei sieht nicht wie ein Python-Interpreter von Nexus aus.\n"
            f"Gefunden: {name}"
        )

    parts_lower = [part.lower() for part in Path(candidate).parts]

    if not any("vicon" in part for part in parts_lower):
        return True, (
            "Pfad existiert, enthält aber keinen offensichtlichen Vicon-Bezug.\n"
            "Bitte manuell prüfen:\n"
            f"{candidate}"
        )

    if not any(part.startswith("nexus") for part in parts_lower):
        return True, (
            "Pfad existiert, aber es wurde kein eindeutiger Nexus-Ordner erkannt.\n"
            "Bitte manuell prüfen:\n"
            f"{candidate}"
        )

    return True, f"Nexus-Python scheint gültig zu sein:\n{candidate}"


def resolve_extract_script_path() -> Path:
    """
    Resolve the extraction script path in both development and bundle modes.

    Priority:
    1) config/vicon_path_config.json -> vicon.extract_script
    2) default extract script path
    """
    vicon_cfg = get_vicon_config()
    configured_script = str(vicon_cfg.get("extract_script", DEFAULT_EXTRACT_SCRIPT)).strip()
    script_rel_path = configured_script or DEFAULT_EXTRACT_SCRIPT

    root = bundle_root_dir()
    candidate = root / script_rel_path
    if candidate.exists():
        return candidate

    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        candidate2 = exe_dir / script_rel_path
        if candidate2.exists():
            return candidate2

    return candidate


def run_extraction(
    log_cb: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Run the Vicon Nexus extraction script using the Nexus Python 2.7 interpreter.

    Returns
    -------
    str | None
        If the extractor prints a line like 'JSON_PATH::<path>', return that path.
        Otherwise return None.
    """
    def log(message: str) -> None:
        if log_cb:
            log_cb(message)
        else:
            print(message)

    python27_path = find_nexus_python27()
    if not python27_path:
        log("FEHLER: Nexus-Python 2.7 wurde nicht gefunden.")
        log("Bitte Pfad in config/vicon_path_config.json prüfen.")
        return None

    script_path = resolve_extract_script_path()

    log(f"[Vicon] Nexus Python gefunden: {python27_path}")
    log(f"[Vicon] Extract-Skript: {script_path}")

    if not os.path.exists(python27_path):
        log(f"FEHLER: Nexus-Python 2.7 wurde nicht gefunden: {python27_path}")
        return None

    if not script_path.exists():
        log(f"FEHLER: Extraktionsskript wurde nicht gefunden: {script_path}")
        return None

    try:
        process = subprocess.Popen(
            [python27_path, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        out, err = process.communicate()
        json_path = None

        if out:
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                log(line)
                if line.startswith("JSON_PATH::"):
                    json_path = line.replace("JSON_PATH::", "").strip()

        if err:
            for line in err.splitlines():
                line = line.strip()
                if line:
                    log(f"FEHLER: {line}")

        if process.returncode != 0:
            log(f"FEHLER: Extraktion fehlgeschlagen (Return-Code: {process.returncode}).")

        return json_path

    except Exception as exc:
        log(f"FEHLER: Extraktion konnte nicht gestartet werden: {exc}")
        return None


if __name__ == "__main__":
    run_extraction()