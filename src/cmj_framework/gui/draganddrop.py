from PySide6.QtCore import Signal, Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect


class JsonDropZone(QLabel):
    """
    Drag & drop zone for CMJ inputs (files or folders).

    Notes
    -----
    - User-visible strings are German.
    - Comments/docstrings are English.
    """
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__("Dateien oder Ordner hier ablegen")
        self.setParent(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)

        self.default_style = """
            QLabel {
                border: 2px dashed #888;
                border-radius: 12px;
                font-size: 16px;
                background-color: #fafafa;
            }
        """
        self.hover_style = """
            QLabel {
                border: 2px dashed #0078d4;
                border-radius: 12px;
                font-size: 16px;
                background-color: #e6f2ff;
            }
        """
        self.setStyleSheet(self.default_style)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.setStyleSheet(self.hover_style)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.default_style)

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p:
                paths.append(p)

        if paths:
            self.filesDropped.emit(paths)
            self.animate_success()

        self.setStyleSheet(self.default_style)
        event.acceptProposedAction()

    def animate_success(self):
        anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        anim.setDuration(300)
        anim.setStartValue(0.4)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._anim = anim