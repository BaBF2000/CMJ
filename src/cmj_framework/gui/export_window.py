import os
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QTextEdit, QTabWidget, QSplitter
)
from PySide6.QtCore import Qt, Signal, QObject, QThread


def infer_patient_from_path(json_path: str) -> str:
    """
    Infer patient name from canonical CMJ folder structure:
    base/patient/session/processed/file.json
    """
    try:
        return Path(json_path).parents[2].name or "Unknown"
    except Exception:
        return "Unknown"


def infer_reports_dir_from_combined_json(json_path: str) -> str:
    """
    Infer canonical reports directory from combined JSON path:
    base/patient/session/processed/file_combined.json -> base/patient/session/reports
    """
    try:
        path_obj = Path(json_path)
        session_dir = path_obj.parent.parent  # processed -> session
        return str(session_dir / "reports")
    except Exception:
        return os.path.join(os.path.dirname(json_path), "reports")


class DropList(QListWidget):
    """
    Custom QListWidget that accepts drag & drop of files or folders.
    Emits a list of dropped paths.
    """
    pathsDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        paths = []
        for url in urls:
            path = url.toLocalFile()
            if path:
                paths.append(path)
        self.pathsDropped.emit(paths)


class ReportWorker(QObject):
    """
    Worker object executed in a QThread to generate a Word report without blocking the GUI.
    """
    log = Signal(str)
    finished = Signal(str)          # emits output path
    failed = Signal(str, str)       # emits (error_message, traceback)

    def __init__(
        self,
        json_path: str,
        patient: str,
        enabled_parameters: list[str],
        parameter_md_path: str,
        phases_md_path: str,
    ):
        super().__init__()
        self.json_path = json_path
        self.patient = patient
        self.enabled_parameters = enabled_parameters
        self.parameter_md_path = parameter_md_path
        self.phases_md_path = phases_md_path

    def run(self):
        try:
            from src.cmj_framework.export.word_report import Report

            self.log.emit("[Export] Word-Bericht wird erstellt…")
            out_path = Report(
                self.json_path,
                patient_override=self.patient,
                enabled_parameters=self.enabled_parameters,
                parameter_md_path=self.parameter_md_path,
                phases_md_path=self.phases_md_path,
            )
            self.log.emit(f"[Export] Fertig: {out_path}")
            self.finished.emit(out_path)

        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(str(e), tb)


class ExportReportView(QWidget):
    """
    GUI panel used to generate the Word report from a combined JSON file.
    """
    logSignal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.selected_json = None
        self._thread = None
        self._worker = None

        self.parameter_md_path = None
        self.phases_md_path = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Word-Bericht exportieren (kombinierte JSON)")
        title.setStyleSheet("font-size:16px; font-weight:600;")
        root.addWidget(title)

        self.list = DropList()
        self.list.pathsDropped.connect(self.on_paths_dropped)
        self.list.itemSelectionChanged.connect(self.on_selection_changed)
        root.addWidget(QLabel("Nur Dateien *_combined.json per Drag & Drop hinzufügen:"))
        root.addWidget(self.list)

        row = QHBoxLayout()
        row.addWidget(QLabel("Patient:"))
        self.patient_edit = QLineEdit()
        self.patient_edit.setPlaceholderText("Patientenname eingeben…")
        row.addWidget(self.patient_edit, 1)
        root.addLayout(row)

        self.sel_label = QLabel("Ausgewählte Datei: —")
        self.sel_label.setStyleSheet("color:#444;")
        root.addWidget(self.sel_label)

        # -----------------------
        # Editable export content
        # -----------------------
        splitter = QSplitter(Qt.Horizontal)

        # Left side: parameter selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Anzuzeigende Parameter im Bericht:"))
        self.param_list = QListWidget()
        left_layout.addWidget(self.param_list)

        param_btns = QHBoxLayout()
        self.btn_check_all = QPushButton("Alle")
        self.btn_uncheck_all = QPushButton("Keine")
        self.btn_remove_unchecked = QPushButton("Deaktivierte entfernen")

        self.btn_check_all.clicked.connect(self.check_all_parameters)
        self.btn_uncheck_all.clicked.connect(self.uncheck_all_parameters)
        self.btn_remove_unchecked.clicked.connect(self.remove_unchecked_parameters)

        param_btns.addWidget(self.btn_check_all)
        param_btns.addWidget(self.btn_uncheck_all)
        param_btns.addWidget(self.btn_remove_unchecked)
        left_layout.addLayout(param_btns)

        splitter.addWidget(left_panel)

        # Right side: markdown editors
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Editierbare Inhalte des Berichts:"))

        self.tabs = QTabWidget()

        self.parameter_editor = QTextEdit()
        self.parameter_editor.setPlaceholderText("parameter.md")
        self.tabs.addTab(self.parameter_editor, "parameter.md")

        self.phases_editor = QTextEdit()
        self.phases_editor.setPlaceholderText("phases.md")
        self.tabs.addTab(self.phases_editor, "phases.md")

        right_layout.addWidget(self.tabs)

        md_btns = QHBoxLayout()
        self.btn_reload_md = QPushButton("Markdown neu laden")
        self.btn_save_md = QPushButton("Markdown speichern")

        self.btn_reload_md.clicked.connect(self.reload_markdowns)
        self.btn_save_md.clicked.connect(self.save_markdowns)

        md_btns.addWidget(self.btn_reload_md)
        md_btns.addWidget(self.btn_save_md)
        md_btns.addStretch(1)
        right_layout.addLayout(md_btns)

        splitter.addWidget(right_panel)
        splitter.setSizes([320, 640])

        root.addWidget(splitter, 1)

        # Buttons
        btns = QHBoxLayout()
        self.btn_generate = QPushButton("Bericht generieren")
        self.btn_generate.clicked.connect(self.generate_report)
        self.btn_generate.setEnabled(False)

        self.btn_open_export = QPushButton("Exportordner öffnen")
        self.btn_open_export.clicked.connect(self.open_export_folder)
        self.btn_open_export.setEnabled(False)

        btns.addWidget(self.btn_generate)
        btns.addWidget(self.btn_open_export)
        btns.addStretch(1)
        root.addLayout(btns)

        self.status = QLabel("Status: Bereit")
        root.addWidget(self.status)

        self._init_export_resources()
        self._load_parameter_definitions()
        self.reload_markdowns()

    def _init_export_resources(self):
        """
        Resolve editable markdown files from the helper module.
        """
        try:
            from src.cmj_framework.export.word_report_helpers import ensure_editable_markdown_file

            self.parameter_md_path = ensure_editable_markdown_file("parameter.md")
            self.phases_md_path = ensure_editable_markdown_file("phases.md")

        except Exception as e:
            self.logSignal.emit(f"[Export] Markdown-Ressourcen konnten nicht initialisiert werden: {e}")

    def _load_parameter_definitions(self):
        """
        Load exportable parameter labels from the helper module.
        """
        self.param_list.clear()

        try:
            from src.cmj_framework.export.word_report_helpers import get_all_export_parameter_labels
            labels = get_all_export_parameter_labels()
        except Exception as e:
            self.logSignal.emit(f"[Export] Parameter konnten nicht geladen werden: {e}")
            labels = []

        for label in labels:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            self.param_list.addItem(item)

    def reload_markdowns(self):
        """
        Reload markdown editor content from disk.
        """
        try:
            if self.parameter_md_path and os.path.exists(self.parameter_md_path):
                with open(self.parameter_md_path, "r", encoding="utf-8") as f:
                    self.parameter_editor.setPlainText(f.read())
            else:
                self.parameter_editor.setPlainText("")

            if self.phases_md_path and os.path.exists(self.phases_md_path):
                with open(self.phases_md_path, "r", encoding="utf-8") as f:
                    self.phases_editor.setPlainText(f.read())
            else:
                self.phases_editor.setPlainText("")

            self.logSignal.emit("[Export] Markdown-Dateien geladen.")

        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Markdown konnte nicht geladen werden:\n{e}")

    def save_markdowns(self):
        """
        Save markdown editor content to disk.
        """
        try:
            if self.parameter_md_path:
                with open(self.parameter_md_path, "w", encoding="utf-8") as f:
                    f.write(self.parameter_editor.toPlainText())

            if self.phases_md_path:
                with open(self.phases_md_path, "w", encoding="utf-8") as f:
                    f.write(self.phases_editor.toPlainText())

            self.logSignal.emit("[Export] Markdown-Dateien gespeichert.")

        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Markdown konnte nicht gespeichert werden:\n{e}")

    def get_enabled_parameters(self) -> list[str]:
        """
        Return checked parameter labels.
        """
        enabled = []
        for i in range(self.param_list.count()):
            item = self.param_list.item(i)
            if item.checkState() == Qt.Checked:
                enabled.append(item.text())
        return enabled

    def check_all_parameters(self):
        for i in range(self.param_list.count()):
            self.param_list.item(i).setCheckState(Qt.Checked)

    def uncheck_all_parameters(self):
        for i in range(self.param_list.count()):
            self.param_list.item(i).setCheckState(Qt.Unchecked)

    def remove_unchecked_parameters(self):
        """
        Remove unchecked items from the UI list.
        Removed items will not appear in the report.
        """
        for i in reversed(range(self.param_list.count())):
            item = self.param_list.item(i)
            if item.checkState() != Qt.Checked:
                self.param_list.takeItem(i)

    def on_paths_dropped(self, paths):
        """
        Filter dropped paths and keep only *_combined.json files.
        If a folder is dropped, search recursively.
        """
        valid = []

        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for filename in files:
                        full_path = os.path.join(root, filename)
                        if full_path.endswith("_combined.json"):
                            valid.append(full_path)
            else:
                if path.endswith("_combined.json"):
                    valid.append(path)

        if not valid:
            QMessageBox.warning(
                self,
                "Keine gültige Datei",
                "Bitte nur Dateien mit der Endung *_combined.json hinzufügen."
            )
            return

        existing = {self.list.item(i).text() for i in range(self.list.count())}

        for full_path in valid:
            if full_path not in existing:
                self.list.addItem(QListWidgetItem(full_path))

        self.list.setCurrentRow(self.list.count() - 1)

    def on_selection_changed(self):
        """
        Update GUI when the selected JSON file changes.
        """
        items = self.list.selectedItems()
        if not items:
            self.selected_json = None
            self.btn_generate.setEnabled(False)
            self.btn_open_export.setEnabled(False)
            self.sel_label.setText("Ausgewählte Datei: —")
            return

        self.selected_json = items[0].text()
        self.sel_label.setText(f"Ausgewählte Datei: {self.selected_json}")

        if not self.patient_edit.text().strip():
            self.patient_edit.setText(infer_patient_from_path(self.selected_json))

        self.btn_generate.setEnabled(True)
        self.btn_open_export.setEnabled(True)

    def generate_report(self):
        """
        Generate the Word report by importing and calling the Report() function directly.
        """
        if not self.selected_json or not os.path.exists(self.selected_json):
            QMessageBox.critical(self, "Fehler", "Die ausgewählte JSON-Datei wurde nicht gefunden.")
            return

        patient = self.patient_edit.text().strip()
        if not patient:
            QMessageBox.warning(self, "Patient fehlt", "Bitte einen Patientennamen eingeben.")
            return

        enabled_parameters = self.get_enabled_parameters()
        if not enabled_parameters:
            QMessageBox.warning(self, "Keine Parameter", "Bitte mindestens einen Parameter auswählen.")
            return

        self.save_markdowns()

        self.btn_generate.setEnabled(False)
        self.btn_open_export.setEnabled(False)
        self.status.setText("Status: Wird ausgeführt…")

        self._thread = QThread(self)
        self._worker = ReportWorker(
            self.selected_json,
            patient,
            enabled_parameters,
            self.parameter_md_path,
            self.phases_md_path,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.logSignal.emit)
        self._worker.finished.connect(self._on_report_finished)
        self._worker.failed.connect(self._on_report_failed)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._cleanup_thread_refs)

        self._thread.start()

    def _cleanup_thread_refs(self):
        """Clear thread references after completion."""
        self._thread = None
        self._worker = None

    def _on_report_finished(self, out_path: str):
        """Handle successful report generation."""
        self.status.setText("Status: Fertig")
        self.btn_generate.setEnabled(True)
        self.btn_open_export.setEnabled(True)

        if sys.platform.startswith("win"):
            try:
                os.startfile(out_path)
            except Exception as e:
                self.logSignal.emit(f"[Export] Konnte Bericht nicht öffnen: {e}")

    def _on_report_failed(self, err_msg: str, tb: str):
        """Handle report generation failure."""
        self.status.setText("Status: Fehler")
        self.btn_generate.setEnabled(True)
        self.btn_open_export.setEnabled(True)

        self.logSignal.emit("[Export] !!! FEHLER !!!")
        self.logSignal.emit(err_msg)
        self.logSignal.emit(tb)

        QMessageBox.critical(
            self,
            "Fehler",
            "Der Bericht konnte nicht erstellt werden.\n"
            "Bitte prüfen Sie die Log-Ausgabe unten."
        )

    def open_export_folder(self):
        """Open the reports folder containing the generated report."""
        if not self.selected_json:
            return

        reports_dir = infer_reports_dir_from_combined_json(self.selected_json)
        os.makedirs(reports_dir, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(reports_dir)

    def set_selected_json(self, json_path: str) -> None:
        """
        Programmatically select a *_combined.json file.
        """
        if not json_path:
            QMessageBox.warning(self, "Keine gültige Datei", "Combined JSON konnte nicht gesetzt werden.")
            return

        json_path = os.path.abspath(json_path)

        if (not os.path.exists(json_path)) or (not json_path.endswith("_combined.json")):
            QMessageBox.warning(self, "Keine gültige Datei", "Combined JSON konnte nicht gesetzt werden.")
            return

        self.list.blockSignals(True)

        existing = [self.list.item(i).text() for i in range(self.list.count())]
        if json_path not in existing:
            self.list.addItem(QListWidgetItem(json_path))

        for i in range(self.list.count()):
            if self.list.item(i).text() == json_path:
                self.list.setCurrentRow(i)
                break

        self.list.blockSignals(False)
        self.on_selection_changed()