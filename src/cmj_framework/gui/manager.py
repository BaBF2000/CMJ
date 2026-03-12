import json
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QDir, QPropertyAnimation, QEasingCurve, QThread, Signal
from PySide6.QtWidgets import (
    QWidget,
    QSplitter,
    QVBoxLayout,
    QPushButton,
    QFrame,
    QTreeView,
    QFileSystemModel,
    QAbstractItemView,
    QProgressBar,
    QTextEdit,
    QMainWindow,
)

from cmj_framework.data_processing.run_processing import (
    process_multiple_json,
    find_cmj_session_dir_from_path,
    build_trial_name_from_json_path,
)
from cmj_framework.data_processing.run_processing_temp_data import TempProcessedData
from cmj_framework.gui.draganddrop import JsonDropZone
from cmj_framework.gui.extraction_window import ExtractionWindow
from cmj_framework.gui.json_formatter import JsonSyntaxHighlighter
from cmj_framework.gui.viewer_widget import CMJViewerWidget
from cmj_framework.utils.pathmanager import PathManager


def find_latest_combined_json(json_paths: list[str]) -> str | None:
    """
    Find the most recent *_combined.json across all inferred session processed folders.
    """
    all_candidates: list[Path] = []

    for path in json_paths:
        session_dir = find_cmj_session_dir_from_path(path)
        if not session_dir:
            continue

        processed_dir = Path(session_dir) / "processed"
        if not processed_dir.is_dir():
            continue

        all_candidates.extend(processed_dir.glob("*_combined.json"))

    if not all_candidates:
        return None

    latest = max(all_candidates, key=lambda path: path.stat().st_mtime)
    return str(latest)


class ProcessingWorker(QThread):
    progress = Signal(int)
    finished = Signal(dict)
    log_signal = Signal(str)

    def __init__(self, json_files: list[str]):
        super().__init__()
        self.json_files = [os.path.abspath(path) for path in json_files]

    def run(self) -> None:
        total = len(self.json_files)
        if total == 0:
            self.log_signal.emit("Keine Dateien zum Verarbeiten gefunden.")
            self.finished.emit({})
            return

        def on_progress(i: int, total_: int) -> None:
            value = int((i / max(1, total_)) * 100)
            self.progress.emit(value)

        def on_log(message: str) -> None:
            self.log_signal.emit(message)

        results, _logs = process_multiple_json(
            self.json_files,
            progress_cb=on_progress,
            log_cb=on_log,
            save_combined=True,
        )
        self.finished.emit(results)


class ViconExporterView(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setStyleSheet("background-color: #f3f3f3;")
        self.parent_main_window = self.get_main_window()

        self.selected_jsons: list[str] = []
        self.processed_results: dict = {}
        self.cmj_viewer: CMJViewerWidget | None = None
        self.worker: ProcessingWorker | None = None
        self.extract_window: ExtractionWindow | None = None
        self.anim: QPropertyAnimation | None = None

        self._last_preview_path: str | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setStyleSheet("QSplitter::handle { background-color: #d0d0d0; }")

        left_panel = self._create_panel_gradient("#ffffff", "#eaeaea")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.setSpacing(15)

        self.btn_extract = self._fluent_button(
            "Daten extrahieren",
            "Bitte stellen Sie sicher, dass Vicon Nexus geöffnet ist.",
        )
        self.btn_visualize = self._fluent_button("Daten visualisieren")

        left_layout.addWidget(self.btn_extract)
        left_layout.addWidget(self.btn_visualize)
        left_layout.addStretch()

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        self.model = QFileSystemModel()

        base_dir = PathManager.canonical_base_dir()
        os.makedirs(base_dir, exist_ok=True)
        PathManager.install_demo_content_to_base_dir(base_dir)
        self._setup_tree(base_dir)

        right_layout.addWidget(self.tree, stretch=8)

        self.json_preview = QTextEdit()
        self.json_preview.setReadOnly(True)
        self.json_preview.setMinimumHeight(180)
        self.json_preview.setProperty("jsonEditor", True)
        self.json_preview.setLineWrapMode(QTextEdit.NoWrap)
        self.json_preview_highlighter = JsonSyntaxHighlighter(self.json_preview.document())
        right_layout.addWidget(self.json_preview)

        self.drop_zone = JsonDropZone()
        self.drop_zone.filesDropped.connect(self.on_jsons_dropped)
        self.drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone.setStyleSheet("border: 2px dashed #888; min-height: 120px;")
        right_layout.addWidget(self.drop_zone, stretch=2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border-radius: 8px;
                background-color: #e0e0e0;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4cc2ff,
                    stop:1 #0078d4
                );
            }
            """
        )
        right_layout.addWidget(self.progress_bar)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 800])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self.btn_extract.clicked.connect(self.open_extraction_window)
        self.btn_visualize.clicked.connect(self.visualize_data)

    # ------------------ UI HELPERS ------------------
    def _create_panel_gradient(self, color_start: str, color_end: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {color_start}, stop:1 {color_end}
                );
                border-radius: 6px;
            }}
            """
        )
        return frame

    def _fluent_button(self, text: str, tooltip: str | None = None) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(50)
        btn.setStyleSheet(
            """
            QPushButton {
                color: #ffffff;
                font-size: 16px;
                border-radius: 12px;
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3c3c3c, stop:1 #2f2f2f
                );
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #505050, stop:1 #424242
                );
            }
            QPushButton:pressed {
                background-color: #2c2c2c;
            }
            """
        )
        if tooltip:
            btn.setToolTip(tooltip)

        def animate() -> None:
            anim = QPropertyAnimation(btn, b"geometry")
            rect = btn.geometry()
            anim.setStartValue(rect)
            anim.setKeyValueAt(0.5, rect.adjusted(-2, -2, 2, 2))
            anim.setEndValue(rect)
            anim.setDuration(150)
            anim.setEasingCurve(QEasingCurve.OutQuad)
            anim.start()
            btn._anim = anim

        btn.pressed.connect(animate)
        return btn

    def _setup_tree(self, root_dir: str) -> None:
        self.model = QFileSystemModel()
        self.model.setRootPath(root_dir)
        self.model.setNameFilters(["*.json"])
        self.model.setNameFilterDisables(False)
        self.model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(root_dir))
        self.tree.setHeaderHidden(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)
        self.tree.setAlternatingRowColors(True)

        self.tree.setDragDropMode(QAbstractItemView.DragOnly)
        self.tree.setDefaultDropAction(Qt.CopyAction)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.tree.doubleClicked.connect(self.open_trial_from_tree)
        self.tree.clicked.connect(self.preview_selected_file)

    # ------------------ MAIN WINDOW LOOKUP ------------------
    def get_main_window(self) -> QMainWindow | None:
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QMainWindow):
                return parent
            parent = parent.parent()
        return None

    # ------------------ PREVIEW HELPERS ------------------
    def set_preview_text(self, text: str) -> None:
        self.json_preview.setPlainText(text)

    def preview_json_file(self, file_path: str) -> None:
        if self._last_preview_path == file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            pretty = json.dumps(data, indent=2, ensure_ascii=False)
            self.json_preview.setPlainText(pretty)
            self._last_preview_path = file_path

        except Exception as exc:
            self.json_preview.setPlainText(f"Fehler beim Lesen der Datei:\n{exc}")
            self._last_preview_path = None

    # ------------------ SESSION KEY HELPERS ------------------
    def extract_session_key(self, file_path: str) -> str | None:
        stem = Path(file_path).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        return f"{parts[0]}_{parts[1]}"

    def group_jsons_by_session_key(self, json_files: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for path in json_files:
            key = self.extract_session_key(path) or "__unknown__"
            groups.setdefault(key, []).append(path)
        return groups

    def choose_best_session(self, groups: dict[str, list[str]]) -> tuple[str | None, list[str], list[str]]:
        if not groups:
            return None, [], []

        def parse_date_from_key(key: str):
            try:
                _, date_str = key.split("_")
                return datetime.strptime(date_str, "%d%m%Y")
            except Exception:
                return datetime.min

        valid_keys = [key for key in groups.keys() if key != "__unknown__"]
        keys_to_consider = valid_keys if valid_keys else list(groups.keys())

        best_key = max(keys_to_consider, key=lambda key: (len(groups[key]), parse_date_from_key(key)))

        kept = groups.get(best_key, [])
        ignored: list[str] = []
        for key, files in groups.items():
            if key != best_key:
                ignored.extend(files)

        return best_key, kept, ignored

    # ------------------ LOGIC ------------------
    def open_extraction_window(self) -> None:
        self.extract_window = ExtractionWindow(self)
        self.extract_window.jsonCreated.connect(self.on_json_created)
        self.extract_window.show()

    def on_json_created(self, json_path: str) -> None:
        directory = os.path.dirname(json_path)
        self.model.setRootPath(directory)
        self.tree.setRootIndex(self.model.index(directory))
        self._last_preview_path = None

    def collect_json_from_paths(self, paths: list[str]) -> list[str]:
        collected: list[str] = []
        for path in paths:
            candidate = Path(path)
            if candidate.is_file() and candidate.suffix.lower() == ".json":
                collected.append(str(candidate))
            elif candidate.is_dir():
                for json_file in candidate.rglob("*_data_cmj.json"):
                    collected.append(str(json_file))
        return collected

    def on_jsons_dropped(self, paths: list[str]) -> None:
        json_files = self.collect_json_from_paths(paths)
        if not json_files:
            self.set_preview_text("Keine CMJ-JSON-Dateien gefunden.")
            return

        groups = self.group_jsons_by_session_key(json_files)
        best_key, kept, ignored = self.choose_best_session(groups)

        if not kept:
            self.set_preview_text("Keine gültigen JSON-Dateien nach Gruppierung gefunden.")
            return

        if ignored:
            msg = (
                f"⚠ Mehrere Sitzungen erkannt. Gewählte Sitzung: '{best_key}' "
                f"({len(kept)} Datei(en)). Ignoriert: {len(ignored)} Datei(en) aus anderen Sitzungen:\n"
                + "\n".join([f"• {Path(path).name}" for path in ignored])
            )
            if self.parent_main_window:
                self.parent_main_window.append_log(msg)

        self.selected_jsons = kept

        directory = str(Path(kept[0]).parent)
        self.model.setRootPath(directory)
        self.tree.setRootIndex(self.model.index(directory))
        self._last_preview_path = None

        preview_text = "📂 CMJ-Sitzung erkannt\n\n"
        preview_text += f"Gewählter Sitzungsschlüssel: {best_key}\n\n"
        preview_text += "Behaltene Dateien:\n"
        for file_path in kept:
            preview_text += f"• {Path(file_path).name}\n"
        preview_text += f"\nGesamtzahl Trials (behalten): {len(kept)}"

        if ignored:
            preview_text += "\n\nWeitere erkannte Sitzungen:\n"
            for key, files in sorted(groups.items(), key=lambda x: x[0]):
                if key == best_key:
                    continue
                preview_text += f"• {key}: {len(files)} Datei(en)\n"

        self.set_preview_text(preview_text)
        self.process_data(kept)

    def preview_selected_file(self, index) -> None:
        if self.model.isDir(index):
            return

        file_path = self.model.filePath(index)
        if not file_path.lower().endswith(".json"):
            return

        self.preview_json_file(file_path)

    def animate_progress(self, target_value: int) -> None:
        self.anim = QPropertyAnimation(self.progress_bar, b"value")
        self.anim.setDuration(300)
        self.anim.setStartValue(self.progress_bar.value())
        self.anim.setEndValue(target_value)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()

    def process_data(self, json_paths: list[str] | None = None) -> None:
        if isinstance(json_paths, bool):
            json_paths = None

        if json_paths is None:
            indexes = self.tree.selectionModel().selectedRows(0)
            json_paths = [self.model.filePath(index) for index in indexes if not self.model.isDir(index)]

        if not json_paths:
            return

        if self.worker is not None and self.worker.isRunning():
            if self.parent_main_window:
                self.parent_main_window.append_log("[INFO] Verarbeitung läuft bereits.")
            return

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("CMJ-Sitzung wird verarbeitet…")

        self.worker = ProcessingWorker(json_paths)
        self.worker.finished.connect(self._cleanup_worker)

        if self.parent_main_window:
            self.worker.log_signal.connect(self.parent_main_window.append_log)

        self.worker.progress.connect(self.animate_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def _cleanup_worker(self, _results: dict) -> None:
        self.worker = None
    
    def shutdown(self) -> None:
        """Wait for running worker threads before widget destruction."""
        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    if self.parent_main_window:
                        self.parent_main_window.append_log("[INFO] Warte auf laufende Verarbeitung vor dem Schließen...")
                    self.worker.wait(3000)
            except Exception:
                pass
            self.worker = None    

    def on_processing_finished(self, results: dict) -> None:
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("✔ Fertig")
        self.processed_results = results

        preview = "📊 Verarbeitung abgeschlossen\n\n"
        for trial_name in results:
            preview += f"• {trial_name}\n"
        preview += f"\nGesamtzahl Trials verarbeitet: {len(results)}"
        self.set_preview_text(preview)

        if not results and self.parent_main_window:
            self.parent_main_window.append_log("[INFO] Keine verarbeitbaren Trials vorhanden.")

        if results:
            viewer = self._open_cmj_viewer_window(refresh=True)
            first_trial = list(results.keys())[0]
            viewer.select_trial(first_trial)

        combined_path = find_latest_combined_json(self.selected_jsons)
        if combined_path:
            if self.parent_main_window and hasattr(self.parent_main_window, "append_log"):
                self.parent_main_window.append_log(f"📄 Combined JSON bereit für Export:\n{combined_path}")

            if self.parent_main_window and hasattr(self.parent_main_window, "open_export_view_with_file"):
                self.parent_main_window.open_export_view_with_file(combined_path)

    def visualize_data(self) -> None:
        indexes = self.tree.selectionModel().selectedRows(0)
        if indexes:
            self.open_trial_from_tree(indexes[0])
            return

        trials = TempProcessedData.list_trials()
        if not trials:
            msg = "Keine Trials im Speicher. Bitte zuerst Daten verarbeiten."
            self.set_preview_text(msg)
            if self.parent_main_window:
                self.parent_main_window.append_log(msg)
            return

        viewer = self._open_cmj_viewer_window(refresh=False)
        viewer.select_trial(trials[0])

    def _open_cmj_viewer_window(self, refresh: bool = False) -> CMJViewerWidget:
        """Create or reuse CMJViewerWidget as a separate window."""
        if self.cmj_viewer is None:
            self.cmj_viewer = CMJViewerWidget()
            self.cmj_viewer.trialRejected.connect(self.on_trial_rejected)
            self.cmj_viewer.setWindowTitle("CMJ Viewer")
            self.cmj_viewer.resize(1280, 800)

        if refresh:
            self.cmj_viewer.refresh_trials()

        if not self.cmj_viewer.isVisible():
            self.cmj_viewer.show()

        self.cmj_viewer.raise_()
        self.cmj_viewer.activateWindow()
        return self.cmj_viewer

    def open_trial_from_tree(self, index) -> None:
        if self.model.isDir(index):
            return

        file_path = self.model.filePath(index)
        if not file_path.lower().endswith(".json"):
            return

        trial_name = build_trial_name_from_json_path(file_path)

        def _tree_log(message: str) -> None:
            if self.parent_main_window:
                self.parent_main_window.append_log(message)

        if trial_name not in TempProcessedData.list_trials():
            try:
                TempProcessedData.load(
                    file_path,
                    trial_name,
                    log_cb=_tree_log,
                    force_reload=False,
                )
                if self.parent_main_window:
                    self.parent_main_window.append_log(f"Trial geladen: {trial_name}")
            except Exception as exc:
                msg = f"Trial '{trial_name}' konnte nicht geladen werden: {exc}"
                self.set_preview_text(msg)
                if self.parent_main_window:
                    self.parent_main_window.append_log(msg)
                return

        viewer = self._open_cmj_viewer_window(refresh=False)
        viewer.select_trial(trial_name)

    def on_trial_rejected(self, old_json_path: str) -> None:
        try:
            session_dir = find_cmj_session_dir_from_path(old_json_path)
            if not session_dir:
                return

            rejected_dir = os.path.join(session_dir, "rejected")
            if not os.path.isdir(rejected_dir):
                return

            self.model.setRootPath(rejected_dir)
            self.tree.setRootIndex(self.model.index(rejected_dir))
            self._last_preview_path = None

            if self.parent_main_window:
                self.parent_main_window.append_log(f"📁 Rejected-Ordner geöffnet: {rejected_dir}")
        except Exception as exc:
            if self.parent_main_window:
                self.parent_main_window.append_log(
                    f"[WARN] Rejected-Ordner konnte nicht geöffnet werden: {exc}"
                )