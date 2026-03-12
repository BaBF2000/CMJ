import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

import importlib
export_window = importlib.import_module("cmj_framework.gui.export_window")


class FakeSignal:
    def __init__(self):
        self._callbacks = []
        self.emitted = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args):
        self.emitted.append(args)
        for callback in self._callbacks:
            callback(*args)


class FakeThread:
    def __init__(self, parent=None):
        self.parent = parent
        self.started = FakeSignal()
        self.finished = FakeSignal()
        self.started_flag = False
        self.quit_called = False
        self.delete_later_called = False

    def start(self):
        self.started_flag = True

    def quit(self):
        self.quit_called = True

    def deleteLater(self):
        self.delete_later_called = True


class FakeWorker:
    def __init__(self, json_path, patient, enabled_parameters, parameter_md_path, phases_md_path):
        self.json_path = json_path
        self.patient = patient
        self.enabled_parameters = enabled_parameters
        self.parameter_md_path = parameter_md_path
        self.phases_md_path = phases_md_path

        self.log = FakeSignal()
        self.finished = FakeSignal()
        self.failed = FakeSignal()

        self.moved_to_thread = None

    def moveToThread(self, thread):
        self.moved_to_thread = thread

    def run(self):
        pass


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def patched_view(monkeypatch, qapp, tmp_path):
    parameter_md = tmp_path / "parameter.md"
    phases_md = tmp_path / "phases.md"
    parameter_md.write_text("parameter content", encoding="utf-8")
    phases_md.write_text("phases content", encoding="utf-8")

    monkeypatch.setattr(
        export_window.ExportReportView,
        "_init_export_resources",
        lambda self: (
            setattr(self, "parameter_md_path", str(parameter_md)),
            setattr(self, "phases_md_path", str(phases_md)),
        ),
    )
    monkeypatch.setattr(
        export_window.ExportReportView,
        "_load_parameter_definitions",
        lambda self: None,
    )

    view = export_window.ExportReportView()
    return view


def test_infer_patient_from_path_returns_patient_name():
    """
    Test that infer_patient_from_path returns the patient folder name.
    """
    path = r"C:\CMJ_manager\Max Mustermann\01.01.2020\processed\session_combined.json"

    result = export_window.infer_patient_from_path(path)

    assert result == "Max Mustermann"


def test_infer_patient_from_path_returns_unknown_for_short_path():
    """
    Test that infer_patient_from_path returns Unknown for invalid paths.
    """
    result = export_window.infer_patient_from_path("session_combined.json")

    assert result == "Unknown"


def test_infer_reports_dir_from_combined_json_returns_reports_dir():
    """
    Test that infer_reports_dir_from_combined_json returns the canonical reports directory.
    """
    path = r"C:\CMJ_manager\Max Mustermann\01.01.2020\processed\session_combined.json"

    result = export_window.infer_reports_dir_from_combined_json(path)

    assert result.endswith(os.path.join("01.01.2020", "reports"))


def test_get_enabled_parameters_returns_only_checked_items(patched_view):
    """
    Test that get_enabled_parameters returns only checked labels.
    """
    view = patched_view

    for label, state in [("A", Qt.Checked), ("B", Qt.Unchecked), ("C", Qt.Checked)]:
        item = export_window.QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(state)
        view.param_list.addItem(item)

    result = view.get_enabled_parameters()

    assert result == ["A", "C"]


def test_check_all_parameters_checks_every_item(patched_view):
    """
    Test that check_all_parameters checks all parameter items.
    """
    view = patched_view

    for label in ["A", "B"]:
        item = export_window.QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Unchecked)
        view.param_list.addItem(item)

    view.check_all_parameters()

    assert view.param_list.item(0).checkState() == Qt.Checked
    assert view.param_list.item(1).checkState() == Qt.Checked


def test_uncheck_all_parameters_unchecks_every_item(patched_view):
    """
    Test that uncheck_all_parameters unchecks all parameter items.
    """
    view = patched_view

    for label in ["A", "B"]:
        item = export_window.QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Checked)
        view.param_list.addItem(item)

    view.uncheck_all_parameters()

    assert view.param_list.item(0).checkState() == Qt.Unchecked
    assert view.param_list.item(1).checkState() == Qt.Unchecked


def test_remove_unchecked_parameters_removes_only_unchecked_items(patched_view):
    """
    Test that remove_unchecked_parameters removes only unchecked items.
    """
    view = patched_view

    for label, state in [("A", Qt.Checked), ("B", Qt.Unchecked), ("C", Qt.Checked)]:
        item = export_window.QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(state)
        view.param_list.addItem(item)

    view.remove_unchecked_parameters()

    assert view.param_list.count() == 2
    assert view.param_list.item(0).text() == "A"
    assert view.param_list.item(1).text() == "C"


def test_reload_markdowns_loads_both_files(patched_view):
    """
    Test that reload_markdowns loads markdown file contents into the editors.
    """
    view = patched_view

    view.reload_markdowns()

    assert "parameter content" in view.parameter_editor.toPlainText()
    assert "phases content" in view.phases_editor.toPlainText()


def test_save_markdowns_writes_editor_content_to_disk(patched_view):
    """
    Test that save_markdowns writes editor content to disk.
    """
    view = patched_view

    view.parameter_editor.setPlainText("new parameter text")
    view.phases_editor.setPlainText("new phases text")

    view.save_markdowns()

    assert Path(view.parameter_md_path).read_text(encoding="utf-8") == "new parameter text"
    assert Path(view.phases_md_path).read_text(encoding="utf-8") == "new phases text"


def test_on_paths_dropped_warns_when_no_valid_combined_json(patched_view, monkeypatch):
    """
    Test that on_paths_dropped warns when no valid combined JSON is found.
    """
    view = patched_view
    warnings = []

    monkeypatch.setattr(
        export_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    view.on_paths_dropped(["C:/tmp/not_valid.json"])

    assert len(warnings) == 1
    assert "Keine gültige Datei" in warnings[0][1]


def test_on_paths_dropped_adds_valid_combined_json_files(patched_view, tmp_path):
    """
    Test that on_paths_dropped keeps only *_combined.json files.
    """
    view = patched_view

    valid1 = tmp_path / "a_combined.json"
    valid2 = tmp_path / "b_combined.json"
    invalid = tmp_path / "c.json"

    valid1.write_text("{}", encoding="utf-8")
    valid2.write_text("{}", encoding="utf-8")
    invalid.write_text("{}", encoding="utf-8")

    view.on_paths_dropped([str(valid1), str(valid2), str(invalid)])

    assert view.list.count() == 2
    assert view.list.item(0).text() == str(valid1)
    assert view.list.item(1).text() == str(valid2)


def test_on_selection_changed_resets_state_when_no_selection(patched_view):
    """
    Test that on_selection_changed resets the UI when nothing is selected.
    """
    view = patched_view

    view.selected_json = "something"
    view.btn_generate.setEnabled(True)
    view.btn_open_export.setEnabled(True)

    view.on_selection_changed()

    assert view.selected_json is None
    assert view.btn_generate.isEnabled() is False
    assert view.btn_open_export.isEnabled() is False
    assert view.sel_label.text() == "Ausgewählte Datei: —"


def test_on_selection_changed_updates_selected_json_and_patient(patched_view, tmp_path):
    """
    Test that on_selection_changed updates selected_json and infers the patient name.
    """
    view = patched_view

    combined = tmp_path / "Max Mustermann" / "01.01.2020" / "processed" / "session_combined.json"
    combined.parent.mkdir(parents=True)
    combined.write_text("{}", encoding="utf-8")

    view.list.addItem(export_window.QListWidgetItem(str(combined)))
    view.list.setCurrentRow(0)

    view.on_selection_changed()

    assert view.selected_json == str(combined)
    assert view.btn_generate.isEnabled() is True
    assert view.btn_open_export.isEnabled() is True
    assert view.patient_edit.text() == "Max Mustermann"


def test_generate_report_warns_when_selected_json_missing(patched_view, monkeypatch):
    """
    Test that generate_report fails when the selected JSON is missing.
    """
    view = patched_view
    criticals = []

    monkeypatch.setattr(
        export_window.QMessageBox,
        "critical",
        lambda *args: criticals.append(args),
    )

    view.selected_json = "C:/missing/session_combined.json"
    view.generate_report()

    assert len(criticals) == 1
    assert "Die ausgewählte JSON-Datei wurde nicht gefunden." in criticals[0][2]


def test_generate_report_warns_when_patient_missing(patched_view, tmp_path, monkeypatch):
    """
    Test that generate_report warns when the patient name is missing.
    """
    view = patched_view
    warnings = []

    combined = tmp_path / "session_combined.json"
    combined.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        export_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    view.selected_json = str(combined)
    view.patient_edit.setText("")
    view.generate_report()

    assert len(warnings) == 1
    assert "Patient fehlt" in warnings[0][1]


def test_generate_report_warns_when_no_parameters_selected(patched_view, tmp_path, monkeypatch):
    """
    Test that generate_report warns when no parameters are selected.
    """
    view = patched_view
    warnings = []

    combined = tmp_path / "session_combined.json"
    combined.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        export_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )
    monkeypatch.setattr(view, "get_enabled_parameters", lambda: [])

    view.selected_json = str(combined)
    view.patient_edit.setText("Max")
    view.generate_report()

    assert len(warnings) == 1
    assert "Keine Parameter" in warnings[0][1]


def test_generate_report_creates_thread_and_worker(patched_view, tmp_path, monkeypatch):
    """
    Test that generate_report creates and starts the export worker thread.
    """
    view = patched_view

    combined = tmp_path / "session_combined.json"
    combined.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(export_window, "QThread", FakeThread)
    monkeypatch.setattr(export_window, "ReportWorker", FakeWorker)
    monkeypatch.setattr(view, "get_enabled_parameters", lambda: ["A", "B"])

    saved = {"called": 0}
    monkeypatch.setattr(view, "save_markdowns", lambda: saved.__setitem__("called", saved["called"] + 1))

    view.selected_json = str(combined)
    view.patient_edit.setText("Max")

    view.generate_report()

    assert saved["called"] == 1
    assert isinstance(view._thread, FakeThread)
    assert isinstance(view._worker, FakeWorker)
    assert view._worker.json_path == str(combined)
    assert view._worker.patient == "Max"
    assert view._worker.enabled_parameters == ["A", "B"]
    assert view._worker.moved_to_thread is view._thread
    assert view._thread.started_flag is True
    assert view.status.text() == "Status: Wird ausgeführt…"
    assert view.btn_generate.isEnabled() is False
    assert view.btn_open_export.isEnabled() is False


def test_cleanup_thread_refs_resets_references(patched_view):
    """
    Test that _cleanup_thread_refs clears thread references.
    """
    view = patched_view
    view._thread = object()
    view._worker = object()

    view._cleanup_thread_refs()

    assert view._thread is None
    assert view._worker is None


def test_on_report_finished_updates_state(patched_view):
    """
    Test that _on_report_finished restores the UI state.
    """
    view = patched_view

    view._on_report_finished("C:/tmp/report.docx")

    assert view.status.text() == "Status: Fertig"
    assert view.btn_generate.isEnabled() is True
    assert view.btn_open_export.isEnabled() is True


def test_on_report_failed_updates_state_and_logs(patched_view):
    """
    Test that _on_report_failed restores the UI and emits logs.
    """
    view = patched_view
    messages = []
    criticals = []

    view.logSignal.connect(messages.append)
    export_window.QMessageBox.critical = lambda *args: criticals.append(args)

    view._on_report_failed("boom", "traceback text")

    assert view.status.text() == "Status: Fehler"
    assert view.btn_generate.isEnabled() is True
    assert view.btn_open_export.isEnabled() is True
    assert messages[0] == "[Export] !!! FEHLER !!!"
    assert messages[1] == "boom"
    assert messages[2] == "traceback text"
    assert len(criticals) == 1


def test_open_export_folder_creates_reports_dir(patched_view, tmp_path, monkeypatch):
    """
    Test that open_export_folder creates the reports directory.
    """
    view = patched_view

    combined = tmp_path / "Max" / "01.01.2020" / "processed" / "session_combined.json"
    combined.parent.mkdir(parents=True)
    combined.write_text("{}", encoding="utf-8")

    view.selected_json = str(combined)

    opened = {"path": None}
    monkeypatch.setattr(export_window.sys, "platform", "win32")
    monkeypatch.setattr(export_window.os, "startfile", lambda path: opened.__setitem__("path", path), raising=False)

    view.open_export_folder()

    expected_dir = combined.parent.parent / "reports"
    assert expected_dir.exists()
    assert opened["path"] == str(expected_dir)


def test_set_selected_json_warns_for_invalid_path(patched_view, monkeypatch):
    """
    Test that set_selected_json warns for invalid input.
    """
    view = patched_view
    warnings = []

    monkeypatch.setattr(
        export_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    view.set_selected_json("C:/missing/not_combined.json")

    assert len(warnings) == 1
    assert "Keine gültige Datei" in warnings[0][1]


def test_set_selected_json_adds_and_selects_valid_file(patched_view, tmp_path):
    """
    Test that set_selected_json adds a valid combined JSON and selects it.
    """
    view = patched_view

    combined = tmp_path / "session_combined.json"
    combined.write_text("{}", encoding="utf-8")

    view.set_selected_json(str(combined))

    assert view.list.count() == 1
    assert os.path.abspath(str(combined)) == view.list.item(0).text()
    assert view.selected_json == os.path.abspath(str(combined))
    assert view.btn_generate.isEnabled() is True
    assert view.btn_open_export.isEnabled() is True