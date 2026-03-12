from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QWidget

from cmj_framework.gui import welcome_window


@pytest.fixture
def qapp():
    """
    Ensure a QApplication exists for widget tests.
    """
    app = welcome_window.QApplication.instance()
    if app is None:
        app = welcome_window.QApplication([])
    return app


@pytest.fixture
def patched_main_window(monkeypatch, qapp, tmp_path):
    """
    Build MainWindow with lightweight fake child views and patched dependencies.
    """
    base_dir = tmp_path / "CMJ_manager"
    base_dir.mkdir()

    monkeypatch.setattr(
        welcome_window.PathManager,
        "canonical_base_dir",
        lambda: str(base_dir),
    )
    monkeypatch.setattr(
        welcome_window.PathManager,
        "install_demo_content_to_base_dir",
        lambda base: None,
    )

    monkeypatch.setattr(
        welcome_window,
        "gui_asset",
        lambda *parts: tmp_path / "missing_asset",
    )

    monkeypatch.setattr(
        welcome_window,
        "documentation_file",
        lambda filename: tmp_path / filename,
    )

    class FakeFolderExplorer(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.root_dir = None

        def set_root_dir(self, root_dir):
            self.root_dir = root_dir

    class FakeSignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, value):
            for callback in self._callbacks:
                callback(value)

    class FakeExportReportView(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.logSignal = FakeSignal()
            self.selected_json = None
            self.shutdown_called = False

        def set_selected_json(self, path):
            self.selected_json = path

        def shutdown(self):
            self.shutdown_called = True

    class FakeParameterView(QWidget):
        def __init__(self, parent=None, log_callback=None):
            super().__init__(parent)
            self.log_callback = log_callback

    class FakeViconExporterView(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.shutdown_called = False

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(welcome_window, "FolderExplorer", FakeFolderExplorer)
    monkeypatch.setattr(welcome_window, "ExportReportView", FakeExportReportView)
    monkeypatch.setattr(welcome_window, "ParameterView", FakeParameterView)
    monkeypatch.setattr(welcome_window, "ViconExporterView", FakeViconExporterView)
    monkeypatch.setattr(welcome_window, "register_gui_logger", lambda callback: None)

    window = welcome_window.MainWindow()
    return window


def test_get_cmj_base_dir_via_pathmanager_creates_base_dir(monkeypatch, tmp_path):
    """
    Test that get_cmj_base_dir_via_pathmanager creates the canonical base dir
    and installs demo content.
    """
    base_dir = tmp_path / "CMJ_manager"
    called = {"install": None}

    monkeypatch.setattr(
        welcome_window.PathManager,
        "canonical_base_dir",
        lambda: str(base_dir),
    )
    monkeypatch.setattr(
        welcome_window.PathManager,
        "install_demo_content_to_base_dir",
        lambda path: called.__setitem__("install", path),
    )

    result = welcome_window.get_cmj_base_dir_via_pathmanager()

    assert result == str(base_dir)
    assert base_dir.exists()
    assert called["install"] == str(base_dir)


def test_append_log_adds_text_to_log_panel(patched_main_window):
    """
    Test that append_log appends text to the log panel.
    """
    window = patched_main_window

    window.append_log("Hello log")

    assert "Hello log" in window.log_panel.toPlainText()


def test_toggle_log_panel_switches_visibility_and_button_text(patched_main_window, qapp):
    """
    Test that toggle_log_panel toggles the panel visibility and button label.
    """
    window = patched_main_window
    window.show()
    qapp.processEvents()

    assert window.log_panel.isVisible() is False
    assert window.btn_toggle_log.text() == "Log anzeigen"

    window.toggle_log_panel()
    qapp.processEvents()
    assert window.log_panel.isVisible() is True
    assert window.btn_toggle_log.text() == "Log ausblenden"

    window.toggle_log_panel()
    qapp.processEvents()
    assert window.log_panel.isVisible() is False
    assert window.btn_toggle_log.text() == "Log anzeigen"


def test_toggle_folder_explorer_switches_visibility_and_button_text(patched_main_window, qapp):
    """
    Test that toggle_folder_explorer toggles the right panel and button text.
    """
    window = patched_main_window
    window.show()
    qapp.processEvents()

    assert window.right_panel.isVisible() is False
    assert window.btn_toggle_files.text() == "Ordner öffnen"

    window.toggle_folder_explorer()
    qapp.processEvents()
    assert window.right_panel.isVisible() is True
    assert window.btn_toggle_files.text() == "Ordner schließen"

    window.toggle_folder_explorer()
    qapp.processEvents()
    assert window.right_panel.isVisible() is False
    assert window.btn_toggle_files.text() == "Ordner öffnen"


def test_show_page_analyse_updates_current_page_and_titles(patched_main_window):
    """
    Test that show_page('analyse') selects the analyse page and updates titles.
    """
    window = patched_main_window

    window.show_page("analyse")

    assert window.pages.currentWidget() is window._page_analyse
    assert window.page_title.text() == "Analyse"
    assert "Importieren" in window.page_subtitle.text()


def test_show_page_export_updates_current_page_and_titles(patched_main_window):
    """
    Test that show_page('export') selects the export page and updates titles.
    """
    window = patched_main_window

    window.show_page("export")

    assert window.pages.currentWidget() is window._page_export
    assert window.page_title.text() == "Word-Export"
    assert "*_combined.json" in window.page_subtitle.text()


def test_show_page_parameter_updates_current_page_and_titles(patched_main_window):
    """
    Test that show_page('parameter') selects the parameter page and updates titles.
    """
    window = patched_main_window

    window.show_page("parameter")

    assert window.pages.currentWidget() is window._page_parameter
    assert window.page_title.text() == "Parameter"
    assert "JSON-Konfigurationen" in window.page_subtitle.text()


def test_show_page_doc_updates_current_page_and_titles(patched_main_window):
    """
    Test that show_page('doc') selects the documentation page and updates titles.
    """
    window = patched_main_window

    window.show_page("doc")

    assert window.pages.currentWidget() is window._page_doc
    assert window.page_title.text() == "Dokumentation"
    assert "Kurzanleitung" in window.page_subtitle.text()


def test_open_export_view_with_file_sets_selected_json_and_logs(patched_main_window):
    """
    Test that open_export_view_with_file switches to export and forwards the file.
    """
    window = patched_main_window
    combined_json = "C:/data/session_combined.json"

    window.open_export_view_with_file(combined_json)

    assert window.pages.currentWidget() is window._page_export
    assert window._page_export.selected_json == combined_json
    assert "Export vorbereitet" in window.log_panel.toPlainText()


def test_open_export_view_with_file_logs_warning_on_failure(patched_main_window):
    """
    Test that open_export_view_with_file logs a warning if set_selected_json fails.
    """
    window = patched_main_window

    def raise_set_selected_json(path):
        raise RuntimeError("cannot set json")

    window.show_page("export")
    window._page_export.set_selected_json = raise_set_selected_json

    window.open_export_view_with_file("C:/data/session_combined.json")

    assert "Konnte combined JSON nicht setzen" in window.log_panel.toPlainText()


def test_open_documentation_logs_warning_when_file_is_missing(patched_main_window, monkeypatch, tmp_path):
    """
    Test that open_documentation logs a warning when the documentation file is missing.
    """
    window = patched_main_window
    monkeypatch.setattr(
        welcome_window,
        "documentation_file",
        lambda filename: tmp_path / "missing_doc.html",
    )

    window.open_documentation()

    assert "Dokumentation nicht gefunden" in window.log_panel.toPlainText()


def test_open_documentation_opens_browser_when_file_exists(patched_main_window, monkeypatch, tmp_path):
    """
    Test that open_documentation opens the browser when the file exists.
    """
    window = patched_main_window
    doc_path = tmp_path / "CMJ_Framework_Documentation.html"
    doc_path.write_text("<html></html>", encoding="utf-8")

    opened = {"url": None}

    monkeypatch.setattr(
        welcome_window,
        "documentation_file",
        lambda filename: doc_path,
    )
    monkeypatch.setattr(
        welcome_window.webbrowser,
        "open",
        lambda url: opened.__setitem__("url", url),
    )

    window.open_documentation()

    assert opened["url"] == doc_path.resolve().as_uri()


def test_load_app_qss_returns_file_content_when_present(monkeypatch, tmp_path):
    """
    Test that load_app_qss returns the stylesheet content when the file exists.
    """
    qss_path = tmp_path / "app.qss"
    qss_path.write_text("QWidget { color: red; }", encoding="utf-8")

    monkeypatch.setattr(welcome_window, "gui_asset", lambda *parts: qss_path)

    result = welcome_window.load_app_qss()

    assert result == "QWidget { color: red; }"


def test_load_app_qss_returns_empty_string_when_missing(monkeypatch, tmp_path):
    """
    Test that load_app_qss returns an empty string when the stylesheet is missing.
    """
    monkeypatch.setattr(
        welcome_window,
        "gui_asset",
        lambda *parts: tmp_path / "missing.qss",
    )

    result = welcome_window.load_app_qss()

    assert result == ""


def test_close_event_calls_shutdown_on_child_views(patched_main_window):
    """
    Test that closeEvent calls shutdown on analyse and export pages when present.
    """
    window = patched_main_window
    window.show_page("analyse")
    window.show_page("export")

    event = SimpleNamespace(accepted=False)

    def accept():
        event.accepted = True

    event.accept = accept

    window.closeEvent(event)

    assert window._page_analyse.shutdown_called is True
    assert window._page_export.shutdown_called is True
    assert event.accepted is True