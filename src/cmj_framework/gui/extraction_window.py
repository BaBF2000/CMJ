import os

from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QObject, QThread


class ExtractionWorker(QObject):
    """
    Worker executed in a QThread to run Vicon extraction without blocking the GUI.
    """

    log = Signal(str)
    finished = Signal(str)  # json_path ("" if none)

    def run(self) -> None:
        from src.cmj_framework.data_processing.run_extraction import run_extraction

        def _log(msg: str) -> None:
            self.log.emit(msg)

        json_path = run_extraction(log_cb=_log) or ""
        self.finished.emit(json_path)


class ExtractionWindow(QWidget):
    """
    Small window to trigger Vicon extraction.

    Notes
    -----
    - User-visible messages are German.
    - Comments/docstrings are English.
    - Emits jsonCreated(json_path) when the extractor returns a JSON path.
    """

    jsonCreated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Vicon-Extraktion")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setFixedSize(260, 180)

        layout = QVBoxLayout(self)

        self.status = QLabel("Bereit")
        self.status.setAlignment(Qt.AlignCenter)

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(90, 90)
        self.play_btn.clicked.connect(self.run_extraction)

        layout.addWidget(self.play_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.status)

        self._thread: QThread | None = None
        self._worker: ExtractionWorker | None = None

    def run_extraction(self) -> None:
        """Start the extraction in a worker thread."""
        self.status.setText("Extraktion läuft…")
        self.play_btn.setEnabled(False)

        self._thread = QThread(self)
        self._worker = ExtractionWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)

        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._cleanup)

        self._thread.start()

    def _cleanup(self) -> None:
        self._thread = None
        self._worker = None

    def _on_log(self, msg: str) -> None:
        if msg.startswith("FEHLER"):
            self.status.setText("Fehler: siehe Log-Ausgabe.")

    def _on_finished(self, json_path: str) -> None:
        if json_path and os.path.exists(json_path):
            self.status.setText("JSON erstellt.")
            self.jsonCreated.emit(json_path)
        else:
            self.status.setText("Fertig (kein JSON gefunden).")

        self.play_btn.setEnabled(True)