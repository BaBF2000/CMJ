import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QTabWidget,
    QMessageBox,
    QFrame,
)

from cmj_framework.data_processing.run_extraction import validate_nexus_python27_path
from cmj_framework.gui.json_formatter import JsonSyntaxHighlighter
from cmj_framework.utils.runtime_paths import config_file


def default_config_path(active_path: Path) -> Path:
    """Return the default config file path for a given active config file."""
    return active_path.with_name(f"{active_path.stem}.default{active_path.suffix}")


class JsonEditorTab(QWidget):
    def __init__(self, title: str, file_path: Path, log_callback=None, parent=None):
        super().__init__(parent)

        self.title = title
        self.file_path = file_path
        self.default_file_path = default_config_path(file_path)
        self.log_callback = log_callback

        root = QVBoxLayout(self)
        root.setSpacing(10)

        info_card = QFrame()
        info_layout = QVBoxLayout(info_card)

        title_label = QLabel(self.title)
        title_label.setObjectName("PageTitle")

        desc = QLabel(
            "Aktive Konfiguration dieser Datei.\n"
            "Änderungen wirken sich nach dem Speichern auf die Anwendung aus.\n\n"
            "Zurücksetzen lädt die Standardwerte.\n"
            "Als Standard speichern ersetzt diese dauerhaft."
        )
        desc.setObjectName("PageSubtitle")
        desc.setWordWrap(True)

        self.path_label = QLabel(f"Aktiv: {self.file_path}")
        self.default_path_label = QLabel(f"Standard: {self.default_file_path}")

        info_layout.addWidget(title_label)
        info_layout.addWidget(desc)
        info_layout.addWidget(self.path_label)
        info_layout.addWidget(self.default_path_label)

        root.addWidget(info_card)

        self.editor = QTextEdit()
        self.editor.setLineWrapMode(QTextEdit.NoWrap)
        self.highlighter = JsonSyntaxHighlighter(self.editor.document())
        root.addWidget(self.editor, 1)

        btn_row = QHBoxLayout()

        self.btn_reload = QPushButton("Neu laden")
        self.btn_format = QPushButton("Formatieren")
        self.btn_save = QPushButton("Speichern")
        self.btn_reset = QPushButton("Zurücksetzen")
        self.btn_replace_default = QPushButton("Als Standard speichern")

        self.btn_reload.clicked.connect(self.load_file)
        self.btn_format.clicked.connect(self.format_json)
        self.btn_save.clicked.connect(self.save_file)
        self.btn_reset.clicked.connect(self.reset_to_default)
        self.btn_replace_default.clicked.connect(self.replace_default_with_current)

        btn_row.addWidget(self.btn_reload)
        btn_row.addWidget(self.btn_format)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_replace_default)

        root.addLayout(btn_row)

        self.load_file()

    def _read_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _write_json(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    def load_file(self):
        if not self.file_path.exists():
            self.editor.setPlainText("")
            self._log(f"[WARN] Konfigurationsdatei nicht gefunden: {self.file_path}")
            return

        data = self._read_json(self.file_path)
        self.editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))

    def format_json(self):
        try:
            raw = self.editor.toPlainText().strip()
            data = json.loads(raw) if raw else {}
            self.editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:
            QMessageBox.warning(self, "Ungültiges JSON", str(exc))

    def save_file(self):
        try:
            raw = self.editor.toPlainText().strip()
            data = json.loads(raw) if raw else {}
            self._write_json(self.file_path, data)
            self._log(f"[Config] Gespeichert: {self.file_path}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Speichern fehlgeschlagen",
                f"Die Datei konnte nicht gespeichert werden:\n\n{exc}",
            )

    def reset_to_default(self, ask_confirmation=True):
        if ask_confirmation:
            reply = QMessageBox.question(self, "Zurücksetzen", "Standardwerte laden?")
            if reply != QMessageBox.Yes:
                return

        if not self.default_file_path.exists():
            QMessageBox.warning(self, "Fehlt", f"Default-Datei nicht gefunden:\n{self.default_file_path}")
            return

        try:
            data = self._read_json(self.default_file_path)
            self._write_json(self.file_path, data)
            self.editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            self._log(f"[Config] Zurückgesetzt auf Standard: {self.file_path}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Zurücksetzen fehlgeschlagen",
                f"Die Standardkonfiguration konnte nicht geladen werden:\n\n{exc}",
            )

    def replace_default_with_current(self):
        try:
            raw = self.editor.toPlainText().strip()
            data = json.loads(raw) if raw else {}
            self._write_json(self.default_file_path, data)
            self._log(f"[Config] Standard überschrieben: {self.default_file_path}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Standard speichern fehlgeschlagen",
                f"Die Standarddatei konnte nicht aktualisiert werden:\n\n{exc}",
            )


class ParameterView(QWidget):
    def __init__(self, parent=None, log_callback=None):
        super().__init__(parent)

        root = QVBoxLayout(self)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tab_word = JsonEditorTab(
            "Word Report Config",
            config_file("word_report_config.json"),
            log_callback=log_callback,
        )

        self.tab_utils = JsonEditorTab(
            "CMJ Utils Config",
            config_file("utils_config.json"),
            log_callback=log_callback,
        )

        self.tab_vicon = JsonEditorTab(
            "Vicon Retrieval Config",
            config_file("nexus_data_retrieval_config.json"),
            log_callback=log_callback,
        )

        self.tab_app = JsonEditorTab(
            "Processing / Vicon Path Config",
            config_file("vicon_path_config.json"),
            log_callback=log_callback,
        )

        self.tabs.addTab(self.tab_word, "Word Report")
        self.tabs.addTab(self.tab_utils, "CMJ Utils")
        self.tabs.addTab(self.tab_vicon, "Vicon Retrieval")
        self.tabs.addTab(self.tab_app, "Processing / Vicon")

        bottom = QHBoxLayout()

        self.btn_reload_all = QPushButton("Alle neu laden")
        self.btn_save_all = QPushButton("Alle speichern")
        self.btn_validate_vicon = QPushButton("Vicon-Pfad prüfen")
        self.btn_reset_all = QPushButton("Alle zurücksetzen")

        bottom.addWidget(self.btn_reload_all)
        bottom.addWidget(self.btn_save_all)
        bottom.addWidget(self.btn_validate_vicon)
        bottom.addStretch()
        bottom.addWidget(self.btn_reset_all)

        root.addLayout(bottom)

        self.btn_reload_all.clicked.connect(self.reload_all)
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_reset_all.clicked.connect(self.reset_all)
        self.btn_validate_vicon.clicked.connect(self.validate_vicon_path)

        self.tabs.currentChanged.connect(self._update_button_states)
        self._update_button_states()

    def _update_button_states(self):
        current = self.tabs.currentWidget()
        is_processing_tab = current is self.tab_app
        self.btn_validate_vicon.setVisible(is_processing_tab)
        self.btn_validate_vicon.setEnabled(is_processing_tab)

    def validate_vicon_path(self):
        try:
            raw = self.tab_app.editor.toPlainText().strip()
            data = json.loads(raw) if raw else {}

            vicon_cfg = data.get("vicon", {})
            path = str(vicon_cfg.get("nexus_python27_path", "")).strip()

            is_valid, message = validate_nexus_python27_path(path)

            if is_valid:
                QMessageBox.information(self, "Vicon-Pfad prüfen", message)
            else:
                QMessageBox.warning(self, "Vicon-Pfad prüfen", message)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Vicon-Pfad prüfen",
                f"Die Konfiguration ist ungültig oder konnte nicht gelesen werden:\n\n{exc}",
            )

    def reload_all(self):
        self.tab_word.load_file()
        self.tab_utils.load_file()
        self.tab_vicon.load_file()
        self.tab_app.load_file()

    def save_all(self):
        self.tab_word.save_file()
        self.tab_utils.save_file()
        self.tab_vicon.save_file()
        self.tab_app.save_file()

    def reset_all(self):
        reply = QMessageBox.question(
            self,
            "Alle zurücksetzen",
            "Alle Konfigurationen zurücksetzen?",
        )

        if reply != QMessageBox.Yes:
            return

        self.tab_word.reset_to_default(False)
        self.tab_utils.reset_to_default(False)
        self.tab_vicon.reset_to_default(False)
        self.tab_app.reset_to_default(False)