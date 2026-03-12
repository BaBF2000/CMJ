from pathlib import Path
import re

import pytest

from cmj_framework.utils import app_logging


def test_timestamp_now_returns_expected_format():
    """
    Test that timestamp_now returns a timestamp string in the expected format.
    """
    result = app_logging.timestamp_now()

    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", result)


def test_dated_log_filename_returns_expected_pattern():
    """
    Test that dated_log_filename builds a timestamped filename.
    """
    result = app_logging.dated_log_filename("session")

    assert re.match(r"^session_\d{8}_\d{6}\.log$", result)


def test_dated_log_filename_supports_custom_extension():
    """
    Test that dated_log_filename supports a custom extension.
    """
    result = app_logging.dated_log_filename("session", ext=".txt")

    assert re.match(r"^session_\d{8}_\d{6}\.txt$", result)


def test_get_log_dir_returns_created_directory(tmp_path, monkeypatch):
    """
    Test that get_log_dir creates and returns the central log directory.
    """
    monkeypatch.setattr(app_logging, "log_dir", lambda: tmp_path / "Log")

    result = app_logging.get_log_dir()

    assert result == tmp_path / "Log"
    assert result.exists()
    assert result.is_dir()


def test_get_log_file_returns_expected_path(tmp_path, monkeypatch):
    """
    Test that get_log_file returns a path inside the log directory.
    """
    monkeypatch.setattr(app_logging, "get_log_dir", lambda: tmp_path / "Log")

    result = app_logging.get_log_file("app.log")

    assert result == tmp_path / "Log" / "app.log"


def test_append_log_line_writes_one_line(tmp_path, monkeypatch):
    """
    Test that append_log_line writes one normalized line to the log file.
    """
    log_file = tmp_path / "app.log"
    monkeypatch.setattr(app_logging, "get_log_file", lambda filename: log_file)

    result = app_logging.append_log_line("app.log", "hello world")

    assert result == log_file
    assert log_file.read_text(encoding="utf-8") == "hello world\n"


def test_append_log_line_strips_extra_trailing_newline(tmp_path, monkeypatch):
    """
    Test that append_log_line avoids duplicating trailing newlines.
    """
    log_file = tmp_path / "app.log"
    monkeypatch.setattr(app_logging, "get_log_file", lambda filename: log_file)

    app_logging.append_log_line("app.log", "hello world\n")

    assert log_file.read_text(encoding="utf-8") == "hello world\n"


def test_append_timestamped_log_line_writes_timestamped_line(tmp_path, monkeypatch):
    """
    Test that append_timestamped_log_line prefixes the message with a timestamp.
    """
    log_file = tmp_path / "app.log"
    monkeypatch.setattr(app_logging, "get_log_file", lambda filename: log_file)
    monkeypatch.setattr(app_logging, "timestamp_now", lambda: "2026-03-11 10:00:00")

    app_logging.append_timestamped_log_line("app.log", "message")

    content = log_file.read_text(encoding="utf-8")
    assert content == "[2026-03-11 10:00:00] message\n"


def test_append_block_writes_block_with_final_newline(tmp_path, monkeypatch):
    """
    Test that append_block writes the block and adds a final newline if missing.
    """
    log_file = tmp_path / "block.log"
    monkeypatch.setattr(app_logging, "get_log_file", lambda filename: log_file)

    result = app_logging.append_block("block.log", "line1\nline2")

    assert result == log_file
    assert log_file.read_text(encoding="utf-8") == "line1\nline2\n"


def test_append_block_preserves_existing_final_newline(tmp_path, monkeypatch):
    """
    Test that append_block does not add an extra newline if one already exists.
    """
    log_file = tmp_path / "block.log"
    monkeypatch.setattr(app_logging, "get_log_file", lambda filename: log_file)

    app_logging.append_block("block.log", "line1\nline2\n")

    assert log_file.read_text(encoding="utf-8") == "line1\nline2\n"


def test_make_session_log_file_returns_timestamped_log_path(tmp_path, monkeypatch):
    """
    Test that make_session_log_file returns a timestamped log path without writing it.
    """
    monkeypatch.setattr(app_logging, "get_log_dir", lambda: tmp_path / "Log")
    monkeypatch.setattr(app_logging, "dated_log_filename", lambda prefix: f"{prefix}_20260311_100000.log")

    result = app_logging.make_session_log_file("session")

    assert result == tmp_path / "Log" / "session_20260311_100000.log"
    assert not result.exists()


def test_append_to_path_creates_parent_and_writes_line(tmp_path):
    """
    Test that append_to_path creates parent directories and writes one line.
    """
    path = tmp_path / "nested" / "logs" / "app.log"

    result = app_logging.append_to_path(path, "hello")

    assert result == path
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "hello\n"


def test_append_to_path_strips_trailing_newline(tmp_path):
    """
    Test that append_to_path normalizes trailing newlines.
    """
    path = tmp_path / "nested" / "logs" / "app.log"

    app_logging.append_to_path(path, "hello\n")

    assert path.read_text(encoding="utf-8") == "hello\n"


def test_append_timestamped_to_path_writes_timestamped_line(tmp_path, monkeypatch):
    """
    Test that append_timestamped_to_path writes a timestamped line.
    """
    path = tmp_path / "nested" / "logs" / "app.log"
    monkeypatch.setattr(app_logging, "timestamp_now", lambda: "2026-03-11 10:00:00")

    app_logging.append_timestamped_to_path(path, "hello")

    assert path.read_text(encoding="utf-8") == "[2026-03-11 10:00:00] hello\n"