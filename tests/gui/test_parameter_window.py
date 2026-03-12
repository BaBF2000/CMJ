import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from cmj_framework.gui import parameter_window


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_config_files(tmp_path):
    files = {}

    for name in [
        "word_report_config.json",
        "utils_config.json",
        "nexus_data_retrieval_config.json",
        "vicon_path_config.json",
    ]:
        path = tmp_path / name
        default_path = tmp_path / f"{path.stem}.default{path.suffix}"

        path.write_text(json.dumps({"name": name, "value": 1}), encoding="utf-8")
        default_path.write_text(json.dumps({"name": name, "value": 99}), encoding="utf-8")

        files[name] = path
        files[f"default::{name}"] = default_path

    return files


def test_default_config_path_returns_expected_default_path(tmp_path):
    """
    Test that default_config_path inserts '.default' before the suffix.
    """
    active = tmp_path / "utils_config.json"

    result = parameter_window.default_config_path(active)

    assert result == tmp_path / "utils_config.default.json"


def test_json_editor_tab_load_file_reads_existing_json(qapp, tmp_path):
    """
    Test that JsonEditorTab loads the JSON file content into the editor.
    """
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")

    tab = parameter_window.JsonEditorTab("Test", path)

    content = tab.editor.toPlainText()
    assert '"a": 1' in content
    assert '"b": 2' in content


def test_json_editor_tab_load_file_logs_warning_when_missing(qapp, tmp_path):
    """
    Test that JsonEditorTab logs a warning when the config file is missing.
    """
    messages = []
    path = tmp_path / "missing.json"

    tab = parameter_window.JsonEditorTab("Test", path, log_callback=messages.append)

    assert tab.editor.toPlainText() == ""
    assert any("Konfigurationsdatei nicht gefunden" in msg for msg in messages)


def test_json_editor_tab_format_json_pretty_prints_valid_json(qapp, tmp_path):
    """
    Test that format_json pretty-prints valid JSON text.
    """
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    tab = parameter_window.JsonEditorTab("Test", path)
    tab.editor.setPlainText('{"b":2,"a":1}')

    tab.format_json()

    content = tab.editor.toPlainText()
    assert '"b": 2' in content
    assert '"a": 1' in content


def test_json_editor_tab_format_json_warns_on_invalid_json(qapp, tmp_path, monkeypatch):
    """
    Test that format_json shows a warning when JSON is invalid.
    """
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    tab = parameter_window.JsonEditorTab("Test", path)
    tab.editor.setPlainText("{invalid json}")

    tab.format_json()

    assert len(warnings) == 1
    assert warnings[0][1] == "Ungültiges JSON"


def test_json_editor_tab_save_file_writes_json_and_logs(qapp, tmp_path):
    """
    Test that save_file writes the current JSON to disk and logs the action.
    """
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    messages = []

    tab = parameter_window.JsonEditorTab("Test", path, log_callback=messages.append)
    tab.editor.setPlainText('{"x": 123}')

    tab.save_file()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"x": 123}
    assert any("Gespeichert" in msg for msg in messages)


def test_json_editor_tab_save_file_shows_critical_on_invalid_json(qapp, tmp_path, monkeypatch):
    """
    Test that save_file shows a critical error when JSON is invalid.
    """
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    criticals = []
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "critical",
        lambda *args: criticals.append(args),
    )

    tab = parameter_window.JsonEditorTab("Test", path)
    tab.editor.setPlainText("{invalid json}")

    tab.save_file()

    assert len(criticals) == 1
    assert "Speichern fehlgeschlagen" in criticals[0][1]


def test_json_editor_tab_reset_to_default_reloads_default_content(qapp, tmp_path):
    """
    Test that reset_to_default replaces the active config with the default config.
    """
    path = tmp_path / "config.json"
    default_path = tmp_path / "config.default.json"

    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    default_path.write_text(json.dumps({"value": 99}), encoding="utf-8")

    messages = []
    tab = parameter_window.JsonEditorTab("Test", path, log_callback=messages.append)

    tab.reset_to_default(ask_confirmation=False)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"value": 99}
    assert '"value": 99' in tab.editor.toPlainText()
    assert any("Zurückgesetzt auf Standard" in msg for msg in messages)


def test_json_editor_tab_reset_to_default_asks_confirmation_and_stops_on_no(qapp, tmp_path, monkeypatch):
    """
    Test that reset_to_default stops when the user declines confirmation.
    """
    path = tmp_path / "config.json"
    default_path = tmp_path / "config.default.json"

    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    default_path.write_text(json.dumps({"value": 99}), encoding="utf-8")

    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "question",
        lambda *args: QMessageBox.No,
    )

    tab = parameter_window.JsonEditorTab("Test", path)
    tab.reset_to_default(ask_confirmation=True)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"value": 1}


def test_json_editor_tab_reset_to_default_warns_when_default_missing(qapp, tmp_path, monkeypatch):
    """
    Test that reset_to_default shows a warning when the default file is missing.
    """
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    tab = parameter_window.JsonEditorTab("Test", path)
    if tab.default_file_path.exists():
        tab.default_file_path.unlink()

    tab.reset_to_default(ask_confirmation=False)

    assert len(warnings) == 1
    assert "Default-Datei nicht gefunden" in warnings[0][2]


def test_json_editor_tab_replace_default_with_current_overwrites_default(qapp, tmp_path):
    """
    Test that replace_default_with_current overwrites the default file.
    """
    path = tmp_path / "config.json"
    default_path = tmp_path / "config.default.json"

    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    default_path.write_text(json.dumps({"value": 99}), encoding="utf-8")

    messages = []
    tab = parameter_window.JsonEditorTab("Test", path, log_callback=messages.append)
    tab.editor.setPlainText('{"value": 123}')

    tab.replace_default_with_current()

    saved = json.loads(default_path.read_text(encoding="utf-8"))
    assert saved == {"value": 123}
    assert any("Standard überschrieben" in msg for msg in messages)


def test_json_editor_tab_replace_default_with_current_shows_critical_on_invalid_json(qapp, tmp_path, monkeypatch):
    """
    Test that replace_default_with_current shows a critical error for invalid JSON.
    """
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    criticals = []
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "critical",
        lambda *args: criticals.append(args),
    )

    tab = parameter_window.JsonEditorTab("Test", path)
    tab.editor.setPlainText("{invalid json}")

    tab.replace_default_with_current()

    assert len(criticals) == 1
    assert "Standard speichern fehlgeschlagen" in criticals[0][1]


def test_parameter_view_update_button_states_only_enables_vicon_on_app_tab(qapp, temp_config_files, monkeypatch):
    """
    Test that the Vicon validation button is only visible/enabled on the app tab.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )

    view = parameter_window.ParameterView()
    view.show()
    qapp.processEvents()

    view.tabs.setCurrentWidget(view.tab_word)
    view._update_button_states()
    qapp.processEvents()
    assert view.btn_validate_vicon.isVisible() is False
    assert view.btn_validate_vicon.isEnabled() is False

    view.tabs.setCurrentWidget(view.tab_app)
    view._update_button_states()
    qapp.processEvents()
    assert view.btn_validate_vicon.isVisible() is True
    assert view.btn_validate_vicon.isEnabled() is True


def test_parameter_view_validate_vicon_path_shows_information_on_valid_path(qapp, temp_config_files, monkeypatch):
    """
    Test that validate_vicon_path shows an information dialog for a valid path.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )

    infos = []
    monkeypatch.setattr(
        parameter_window,
        "validate_nexus_python27_path",
        lambda path: (True, "valid path"),
    )
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "information",
        lambda *args: infos.append(args),
    )

    view = parameter_window.ParameterView()
    view.tab_app.editor.setPlainText(
        json.dumps({"vicon": {"nexus_python27_path": "C:/Vicon/python.exe"}})
    )

    view.validate_vicon_path()

    assert len(infos) == 1
    assert infos[0][1] == "Vicon-Pfad prüfen"
    assert infos[0][2] == "valid path"


def test_parameter_view_validate_vicon_path_shows_warning_on_invalid_path(qapp, temp_config_files, monkeypatch):
    """
    Test that validate_vicon_path shows a warning dialog for an invalid path.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )

    warnings = []
    monkeypatch.setattr(
        parameter_window,
        "validate_nexus_python27_path",
        lambda path: (False, "invalid path"),
    )
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    view = parameter_window.ParameterView()
    view.tab_app.editor.setPlainText(
        json.dumps({"vicon": {"nexus_python27_path": "C:/Bad/python.exe"}})
    )

    view.validate_vicon_path()

    assert len(warnings) == 1
    assert warnings[0][1] == "Vicon-Pfad prüfen"
    assert warnings[0][2] == "invalid path"


def test_parameter_view_validate_vicon_path_shows_critical_on_invalid_config(qapp, temp_config_files, monkeypatch):
    """
    Test that validate_vicon_path shows a critical error when the config cannot be parsed.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )

    criticals = []
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "critical",
        lambda *args: criticals.append(args),
    )

    view = parameter_window.ParameterView()
    view.tab_app.editor.setPlainText("{invalid json}")

    view.validate_vicon_path()

    assert len(criticals) == 1
    assert criticals[0][1] == "Vicon-Pfad prüfen"


def test_parameter_view_reload_all_calls_all_tabs(qapp, temp_config_files, monkeypatch):
    """
    Test that reload_all calls load_file on all tabs.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )

    view = parameter_window.ParameterView()

    calls = []
    view.tab_word.load_file = lambda: calls.append("word")
    view.tab_utils.load_file = lambda: calls.append("utils")
    view.tab_vicon.load_file = lambda: calls.append("vicon")
    view.tab_app.load_file = lambda: calls.append("app")

    view.reload_all()

    assert calls == ["word", "utils", "vicon", "app"]


def test_parameter_view_save_all_calls_all_tabs(qapp, temp_config_files, monkeypatch):
    """
    Test that save_all calls save_file on all tabs.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )

    view = parameter_window.ParameterView()

    calls = []
    view.tab_word.save_file = lambda: calls.append("word")
    view.tab_utils.save_file = lambda: calls.append("utils")
    view.tab_vicon.save_file = lambda: calls.append("vicon")
    view.tab_app.save_file = lambda: calls.append("app")

    view.save_all()

    assert calls == ["word", "utils", "vicon", "app"]


def test_parameter_view_reset_all_stops_when_user_declines(qapp, temp_config_files, monkeypatch):
    """
    Test that reset_all does nothing when the user declines confirmation.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "question",
        lambda *args: QMessageBox.No,
    )

    view = parameter_window.ParameterView()

    calls = []
    view.tab_word.reset_to_default = lambda ask_confirmation=False: calls.append("word")
    view.tab_utils.reset_to_default = lambda ask_confirmation=False: calls.append("utils")
    view.tab_vicon.reset_to_default = lambda ask_confirmation=False: calls.append("vicon")
    view.tab_app.reset_to_default = lambda ask_confirmation=False: calls.append("app")

    view.reset_all()

    assert calls == []


def test_parameter_view_reset_all_calls_all_tabs_when_confirmed(qapp, temp_config_files, monkeypatch):
    """
    Test that reset_all resets all tabs when the user confirms.
    """
    monkeypatch.setattr(
        parameter_window,
        "config_file",
        lambda name: temp_config_files[name],
    )
    monkeypatch.setattr(
        parameter_window.QMessageBox,
        "question",
        lambda *args: QMessageBox.Yes,
    )

    view = parameter_window.ParameterView()

    calls = []
    view.tab_word.reset_to_default = lambda ask_confirmation=False: calls.append(("word", ask_confirmation))
    view.tab_utils.reset_to_default = lambda ask_confirmation=False: calls.append(("utils", ask_confirmation))
    view.tab_vicon.reset_to_default = lambda ask_confirmation=False: calls.append(("vicon", ask_confirmation))
    view.tab_app.reset_to_default = lambda ask_confirmation=False: calls.append(("app", ask_confirmation))

    view.reset_all()

    assert calls == [
        ("word", False),
        ("utils", False),
        ("vicon", False),
        ("app", False),
    ]