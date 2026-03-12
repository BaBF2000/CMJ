import os,sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QTreeView,
    QLabel,
    QMessageBox,
    QApplication,
    QFileSystemModel,
    QStyle,
)
from PySide6.QtCore import Qt, QDir, QSettings

# Ensure the project root is importable (dev only; avoid polluting sys.path in PyInstaller bundles)
if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


from src.cmj_framework.utils.pathmanager import PathManager
from src.cmj_framework.gui.json_formatter import JsonPreviewDialog


def get_default_cmj_base_dir() -> str:
    """
    Resolve the canonical CMJ base directory.
    """
    return PathManager.canonical_base_dir()


class FolderExplorer(QWidget):
    """
    Folder explorer widget:
    - Back button (history + fallback to parent directory navigation)
    - Copy path button
    - Reset button (return to canonical CMJ_manager folder)
    - Search bar (filters only the current directory)
    - Tree view
    - JSON preview dialog on double-click
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("FolderExplorer")

        # ===== SETTINGS =====
        self.settings = QSettings("SessionManager", "FolderExplorer")

        default_dir = get_default_cmj_base_dir()
        if not os.path.isdir(default_dir):
            default_dir = os.path.expanduser("~")

        last_dir = self.settings.value("last_dir", "", type=str) or ""

        if last_dir and os.path.isdir(last_dir):
            self.current_dir = last_dir
        else:
            self.current_dir = default_dir

        self.default_dir = default_dir

        # ===== HISTORY =====
        self.history: list[str] = []
        self.history_index = -1

        # Keep preview dialogs alive while open
        self._open_dialogs = []

        # ===== LAYOUT =====
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # ===== TOP BAR =====
        top_bar_widget = QWidget()
        top_bar_widget.setObjectName("FolderExplorerTopBar")
        top_bar = QHBoxLayout(top_bar_widget)
        top_bar.setSpacing(6)
        top_bar.setContentsMargins(0, 0, 0, 0)

        self.btn_return = QPushButton("⏴")
        self.btn_copy_path = QPushButton("📋")
        self.btn_reset = QPushButton()

        self.btn_return.setObjectName("IconButton")
        self.btn_copy_path.setObjectName("IconButton")
        self.btn_reset.setObjectName("IconButton")

        self.btn_return.setToolTip("Zurück")
        self.btn_copy_path.setToolTip("Pfad kopieren")
        self.btn_reset.setToolTip("Zurücksetzen (CMJ_manager)")

        for button in (self.btn_return, self.btn_copy_path, self.btn_reset):
            button.setFixedSize(32, 32)

        self.search_bar = QLineEdit()
        self.search_bar.setObjectName("SearchBar")
        self.search_bar.setPlaceholderText("Ordner oder JSON suchen…")
        self.search_bar.setClearButtonEnabled(True)

        reset_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.btn_reset.setIcon(reset_icon)
        self.btn_reset.setIconSize(self.btn_reset.size() * 0.6)

        top_bar.addWidget(self.btn_return)
        top_bar.addWidget(self.btn_copy_path)
        top_bar.addWidget(self.btn_reset)
        top_bar.addWidget(self.search_bar, 1)

        # ===== PATH LABEL =====
        self.path_label = QLabel(self.current_dir)
        self.path_label.setObjectName("FolderPathLabel")
        self.path_label.setWordWrap(True)

        # ===== FILE SYSTEM MODEL =====
        self.model = QFileSystemModel()
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.model.setNameFilters(["*.json"])
        self.model.setNameFilterDisables(False)
        self.model.setRootPath(self.current_dir)

        self.tree = QTreeView()
        self.tree.setObjectName("FolderTree")
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(self.current_dir))
        self.tree.setHeaderHidden(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(18)
        self.tree.setSortingEnabled(True)
        self.tree.setAlternatingRowColors(True)

        # Hide unused columns
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)

        # Drag & drop
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDefaultDropAction(Qt.CopyAction)

        # ===== ASSEMBLY =====
        layout.addWidget(top_bar_widget)
        layout.addWidget(self.path_label)
        layout.addWidget(self.tree, 1)

        # ===== CONNECTIONS =====
        self.search_bar.textChanged.connect(self.filter_items)
        self.tree.doubleClicked.connect(self.open_item)

        self.btn_return.clicked.connect(self.go_back)
        self.btn_copy_path.clicked.connect(self.copy_path)
        self.btn_reset.clicked.connect(self.reset_to_default)

    # ---------- HELPERS ----------
    def _set_dir(self, path: str) -> None:
        """Update explorer directory."""
        if not path or not os.path.isdir(path):
            return

        self.current_dir = path
        self.path_label.setText(path)

        self.model.setRootPath(path)
        self.tree.setRootIndex(self.model.index(path))

        self.settings.setValue("last_dir", path)

        if self.search_bar.text():
            self.search_bar.blockSignals(True)
            self.search_bar.setText("")
            self.search_bar.blockSignals(False)

    def _push_history(self, previous_dir: str) -> None:
        """Push folder into navigation history."""
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]

        self.history.append(previous_dir)
        self.history_index += 1

    # ---------- FILTER ----------
    def filter_items(self, text: str) -> None:
        text = (text or "").lower().strip()
        root = self.tree.rootIndex()

        rows = self.model.rowCount(root)
        for i in range(rows):
            child = self.model.index(i, 0, root)
            path = self.model.filePath(child)
            name = os.path.basename(path).lower()

            visible = True if not text else (text in name)
            self.tree.setRowHidden(i, root, not visible)

    # ---------- OPEN ----------
    def open_item(self, index) -> None:
        path = self.model.filePath(index)

        if os.path.isdir(path):
            self._push_history(self.current_dir)
            self._set_dir(path)
            return

        if path.lower().endswith(".json"):
            try:
                dialog = JsonPreviewDialog(path, parent=self)
                self._open_dialogs.append(dialog)
                dialog.finished.connect(lambda _: self._cleanup_dialog(dialog))
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()

            except Exception as e:
                QMessageBox.critical(self, "Fehler", str(e))

    def _cleanup_dialog(self, dialog) -> None:
        """Remove closed preview dialog from the internal list."""
        try:
            self._open_dialogs.remove(dialog)
        except ValueError:
            pass

    # ---------- NAVIGATION ----------
    def go_back(self) -> None:
        """Go back in history or parent folder."""
        if self.history_index >= 0:
            prev_dir = self.history[self.history_index]
            self.history_index -= 1
            self._set_dir(prev_dir)
            return

        cur = Path(self.current_dir)
        parent = cur.parent

        if parent == cur:
            return

        self._set_dir(str(parent))

    def reset_to_default(self) -> None:
        """Reset explorer to canonical CMJ_manager folder."""
        self.history = []
        self.history_index = -1
        self._set_dir(self.default_dir)

    def copy_path(self) -> None:
        """Copy selected path or current directory."""
        path_to_copy = None

        selected_rows = self.tree.selectionModel().selectedRows(0)
        if selected_rows:
            path_to_copy = self.model.filePath(selected_rows[0])

        if not path_to_copy:
            idx = self.tree.currentIndex()
            if idx.isValid():
                path_to_copy = self.model.filePath(idx)

        if not path_to_copy:
            path_to_copy = self.current_dir

        QApplication.clipboard().setText(path_to_copy)
        self.btn_copy_path.setToolTip(f"Kopiert: {path_to_copy}")

    def set_root_dir(self, path: str) -> None:
        """Set root directory and reset history."""
        self.history = []
        self.history_index = -1

        if path and os.path.isdir(path):
            self.default_dir = path
            self._set_dir(path)
        else:
            self._set_dir(self.default_dir)