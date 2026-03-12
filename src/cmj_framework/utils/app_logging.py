from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.cmj_framework.utils.runtime_paths import ensure_dir, log_dir


def timestamp_now() -> str:
    """Return current timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dated_log_filename(prefix: str, ext: str = ".log") -> str:
    """Return a timestamped log filename."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}{ext}"


def get_log_dir() -> Path:
    """Return and create the central log directory."""
    return ensure_dir(log_dir())


def get_log_file(filename: str) -> Path:
    """Return absolute path for a log file inside the central log directory."""
    return get_log_dir() / filename


def append_log_line(filename: str, message: str) -> Path:
    """
    Append one line to a log file and return the file path.
    """
    path = get_log_file(filename)
    with open(path, "a", encoding="utf-8") as file:
        file.write(message.rstrip("\n") + "\n")
    return path


def append_timestamped_log_line(filename: str, message: str) -> Path:
    """
    Append one timestamped line to a log file and return the file path.
    """
    return append_log_line(filename, f"[{timestamp_now()}] {message}")


def append_block(filename: str, block: str) -> Path:
    """
    Append a raw text block to a log file and return the file path.
    """
    path = get_log_file(filename)
    with open(path, "a", encoding="utf-8") as file:
        file.write(block)
        if not block.endswith("\n"):
            file.write("\n")
    return path


def make_session_log_file(prefix: str) -> Path:
    """
    Create a timestamped session log path.
    The file is not written immediately.
    """
    return get_log_file(dated_log_filename(prefix))


def append_to_path(path: Path, message: str) -> Path:
    """
    Append one line to a specific path and return it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(message.rstrip("\n") + "\n")
    return path


def append_timestamped_to_path(path: Path, message: str) -> Path:
    """
    Append one timestamped line to a specific path and return it.
    """
    return append_to_path(path, f"[{timestamp_now()}] {message}")