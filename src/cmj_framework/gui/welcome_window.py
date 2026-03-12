import os
import sys
import webbrowser
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QStackedWidget,
    QSizePolicy,
)
from PySide6.QtGui import QIcon

# Ensure the project root is importable in development mode
if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from cmj_framework.data_processing.run_error_handler import register_gui_logger
from cmj_framework.gui.export_window import ExportReportView
from cmj_framework.gui.folder_explorer import FolderExplorer
from cmj_framework.gui.manager import ViconExporterView
from cmj_framework.gui.parameter_window import ParameterView
from cmj_framework.utils.pathmanager import PathManager
from cmj_framework.utils.runtime_paths import (
    documentation_file,
    gui_asset,
)


def get_cmj_base_dir_via_pathmanager() -> str:
    """
    Return the canonical CMJ base directory via PathManager and ensure
    bundled demo content is available there.
    """
    base_dir = PathManager.canonical_base_dir()
    os.makedirs(base_dir, exist_ok=True)
    PathManager.install_demo_content_to_base_dir(base_dir)
    return base_dir


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CMJ Manager")


        self.resize(1120, 780)

        # ----- Root central widget -----
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # =========================
        # Main area: Sidebar + Content + Right dock panel
        # =========================
        main_area = QWidget()
        main_row = QHBoxLayout(main_area)
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)

        # ----- Sidebar -----
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(260)

        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(14, 14, 14, 14)
        sb.setSpacing(10)

        brand = QLabel("CMJ Manager")
        brand.setObjectName("PageTitle")
        sb.addWidget(brand)

        subtitle = QLabel("Klinische Bewegungsanalyse")
        subtitle.setObjectName("PageSubtitle")
        sb.addWidget(subtitle)

        sb.addSpacing(10)

        self.btn_nav_analyse = QPushButton("Analyse")
        self.btn_nav_export = QPushButton("Word-Export")
        self.btn_nav_parameter = QPushButton("Parameter")
        self.btn_nav_doc = QPushButton("Dokumentation")

        for button in (
            self.btn_nav_analyse,
            self.btn_nav_export,
            self.btn_nav_parameter,
            self.btn_nav_doc,
        ):
            button.setObjectName("NavButton")
            button.setProperty("active", False)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        sb.addWidget(self.btn_nav_analyse)
        sb.addWidget(self.btn_nav_export)
        sb.addWidget(self.btn_nav_parameter)
        sb.addWidget(self.btn_nav_doc)

        sb.addStretch(1)

        self.btn_toggle_log = QPushButton("Log anzeigen")
        self.btn_toggle_log.setObjectName("NavButton")
        self.btn_toggle_log.setProperty("active", False)
        sb.addWidget(self.btn_toggle_log)

        main_row.addWidget(self.sidebar)

        # ----- Content column -----
        self.content_col = QWidget()
        content_layout = QVBoxLayout(self.content_col)
        self.central_layout = content_layout
        content_layout.setContentsMargins(16, 14, 16, 12)
        content_layout.setSpacing(10)

        self.header = QWidget()
        self.header.setObjectName("HeaderBar")
        header_row = QHBoxLayout(self.header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)

        self.page_title = QLabel("Analyse")
        self.page_title.setObjectName("PageTitle")
        self.page_subtitle = QLabel("Importieren, verarbeiten, prüfen und Trials bewerten.")
        self.page_subtitle.setObjectName("PageSubtitle")

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)
        title_col.addWidget(self.page_title)
        title_col.addWidget(self.page_subtitle)

        header_row.addLayout(title_col)
        header_row.addStretch(1)

        self.btn_toggle_files = QPushButton("Ordner öffnen")
        self.btn_toggle_files.setObjectName("Secondary")
        header_row.addWidget(self.btn_toggle_files)

        content_layout.addWidget(self.header)

        self.pages = QStackedWidget()
        self.pages.setObjectName("Pages")
        content_layout.addWidget(self.pages, 1)

        self._page_analyse: QWidget | None = None
        self._page_export: QWidget | None = None
        self._page_parameter: QWidget | None = None
        self._page_doc: QWidget | None = None

        main_row.addWidget(self.content_col, 1)

        # =========================
        # Right dock panel
        # =========================
        self.right_panel = QWidget()
        self.right_panel.setObjectName("RightPanel")
        self.right_panel.setFixedWidth(390)

        rp = QVBoxLayout(self.right_panel)
        rp.setContentsMargins(10, 10, 10, 10)
        rp.setSpacing(10)

        self.folder_explorer = FolderExplorer(parent=self.right_panel)
        base_dir = get_cmj_base_dir_via_pathmanager()
        self.folder_explorer.set_root_dir(base_dir)

        rp.addWidget(self.folder_explorer, 1)
        self.right_panel.setVisible(False)
        main_row.addWidget(self.right_panel)

        root_layout.addWidget(main_area, 1)

        # =========================
        # Log panel
        # =========================
        self.log_panel = QTextEdit()
        self.log_panel.setObjectName("LogPanel")
        self.log_panel.setReadOnly(True)
        self.log_panel.setFixedHeight(170)
        self.log_panel.setVisible(False)
        root_layout.addWidget(self.log_panel)

        register_gui_logger(self.append_log)

        # =========================
        # Connections
        # =========================
        self.btn_nav_analyse.clicked.connect(lambda: self.show_page("analyse"))
        self.btn_nav_export.clicked.connect(lambda: self.show_page("export"))
        self.btn_nav_parameter.clicked.connect(lambda: self.show_page("parameter"))
        self.btn_nav_doc.clicked.connect(lambda: self.show_page("doc"))

        self.btn_toggle_files.clicked.connect(self.toggle_folder_explorer)
        self.btn_toggle_log.clicked.connect(self.toggle_log_panel)

        self.show_page("analyse")

    # -------------------------
    # Logging
    # -------------------------
    def append_log(self, message: str) -> None:
        self.log_panel.append(message)

    def toggle_log_panel(self) -> None:
        visible = not self.log_panel.isVisible()
        self.log_panel.setVisible(visible)
        self.btn_toggle_log.setText("Log ausblenden" if visible else "Log anzeigen")

    # -------------------------
    # Sidebar active state
    # -------------------------
    def _set_active_nav(self, which: str) -> None:
        mapping = {
            "analyse": self.btn_nav_analyse,
            "export": self.btn_nav_export,
            "parameter": self.btn_nav_parameter,
            "doc": self.btn_nav_doc,
        }
        for key, btn in mapping.items():
            btn.setProperty("active", key == which)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # -------------------------
    # Lazy pages
    # -------------------------
    def _ensure_analyse_page(self) -> QWidget:
        if self._page_analyse is None:
            self._page_analyse = ViconExporterView(parent=self)
            self.pages.addWidget(self._page_analyse)
        return self._page_analyse

    def _ensure_export_page(self) -> QWidget:
        if self._page_export is None:
            self._page_export = ExportReportView(parent=self)
            self._page_export.logSignal.connect(self.append_log)
            self.pages.addWidget(self._page_export)
        return self._page_export

    def _ensure_parameter_page(self) -> QWidget:
        if self._page_parameter is None:
            self._page_parameter = ParameterView(
                parent=self,
                log_callback=self.append_log,
            )
            self.pages.addWidget(self._page_parameter)
        return self._page_parameter

    def _ensure_doc_page(self) -> QWidget:
        if self._page_doc is None:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)

            title = QLabel("Dokumentation")
            title.setObjectName("PageTitle")
            layout.addWidget(title)

            btn_open_help = QPushButton("Hilfe im Browser öffnen")
            btn_open_help.setObjectName("Secondary")
            btn_open_help.clicked.connect(self.open_documentation)
            layout.addWidget(btn_open_help)

            info = QLabel(
                "Ablauf der Analyse\n\n"
                "1) Daten extrahieren (Vicon)\n"
                "2) Daten verarbeiten\n"
                "3) Trials prüfen (Viewer)\n"
                "4) Report exportieren\n\n"
                "Tipp: Über 'Dateien öffnen' kannst du Pfade kopieren und JSON prüfen."
            )
            info.setObjectName("PageSubtitle")
            info.setWordWrap(True)
            layout.addWidget(info)
            layout.addStretch(1)

            self._page_doc = widget
            self.pages.addWidget(self._page_doc)
        return self._page_doc

    # -------------------------
    # Navigation
    # -------------------------
    def show_page(self, which: str) -> None:
        self._set_active_nav(which)

        if which == "analyse":
            page = self._ensure_analyse_page()
            self.pages.setCurrentWidget(page)
            self.page_title.setText("Analyse")
            self.page_subtitle.setText("Importieren, verarbeiten, prüfen und Trials bewerten.")
            return

        if which == "export":
            page = self._ensure_export_page()
            self.pages.setCurrentWidget(page)
            self.page_title.setText("Word-Export")
            self.page_subtitle.setText("Bericht aus *_combined.json erstellen.")
            return

        if which == "parameter":
            page = self._ensure_parameter_page()
            self.pages.setCurrentWidget(page)
            self.page_title.setText("Parameter")
            self.page_subtitle.setText("JSON-Konfigurationen anzeigen und bearbeiten.")
            return

        if which == "doc":
            page = self._ensure_doc_page()
            self.pages.setCurrentWidget(page)
            self.page_title.setText("Dokumentation")
            self.page_subtitle.setText("Kurzanleitung für klinische Nutzung.")
            return

    def open_export_view_with_file(self, combined_json_path: str) -> None:
        self.show_page("export")
        view = self._ensure_export_page()

        try:
            view.set_selected_json(combined_json_path)
            self.append_log(f"📄 Export vorbereitet:\n{combined_json_path}")
        except Exception as exc:
            self.append_log(f"[WARN] Konnte combined JSON nicht setzen: {exc}")

    # -------------------------
    # Folder explorer dock
    # -------------------------
    def toggle_folder_explorer(self) -> None:
        visible = not self.right_panel.isVisible()
        self.right_panel.setVisible(visible)
        self.btn_toggle_files.setText("Ordner schließen" if visible else "Ordner öffnen")

    def closeEvent(self, event) -> None:
        """Ensure child views shut down cleanly before closing."""
        try:
            if self._page_analyse is not None and hasattr(self._page_analyse, "shutdown"):
                self._page_analyse.shutdown()
        except Exception:
            pass

        try:
            if self._page_export is not None and hasattr(self._page_export, "shutdown"):
                self._page_export.shutdown()
        except Exception:
            pass

        event.accept()

    def open_documentation(self) -> None:
        """Open the HTML documentation in the default browser."""
        doc_path = documentation_file("CMJ_Framework_Documentation.html")

        if not doc_path.exists():
            self.append_log(f"[WARN] Dokumentation nicht gefunden: {doc_path}")
            return

        try:
            webbrowser.open(doc_path.resolve().as_uri())
        except Exception as exc:
            self.append_log(f"[WARN] Dokumentation konnte nicht geöffnet werden: {exc}")


def load_app_qss() -> str:
    qss_path = gui_asset("styles", "app.qss")
    try:
        if qss_path.exists():
            return qss_path.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def main() -> int:
    app = QApplication(sys.argv)

    icon_path = gui_asset("icons", "cmj_logo.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    else:
        print(f"[WARN] Icon not found: {icon_path}")

    app.setStyleSheet(load_app_qss())

    window = MainWindow()
    window.show()

    return int(app.exec())