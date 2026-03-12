import json
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QApplication,
)
from PySide6.QtGui import (
    QTextCharFormat,
    QColor,
    QFont,
    QSyntaxHighlighter,
)
from PySide6.QtCore import QRegularExpression


class JsonSyntaxHighlighter(QSyntaxHighlighter):
    """
    Simple JSON syntax highlighter for QTextEdit.
    """

    def __init__(self, document):
        super().__init__(document)
        self.rules = []

        key_format = QTextCharFormat()
        key_format.setForeground(QColor("#1d4ed8"))
        key_format.setFontWeight(QFont.Bold)

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#047857"))

        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#b45309"))

        bool_format = QTextCharFormat()
        bool_format.setForeground(QColor("#7c3aed"))
        bool_format.setFontWeight(QFont.Bold)

        null_format = QTextCharFormat()
        null_format.setForeground(QColor("#dc2626"))
        null_format.setFontItalic(True)

        brace_format = QTextCharFormat()
        brace_format.setForeground(QColor("#475569"))
        brace_format.setFontWeight(QFont.Bold)

        # JSON keys: "key":
        self.rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"\s*(?=:)'), key_format))

        # JSON string values
        self.rules.append((QRegularExpression(r':\s*"[^"\\]*(\\.[^"\\]*)*"'), string_format))

        # Numbers
        self.rules.append(
            (QRegularExpression(r'\b-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\b'), number_format)
        )

        # true / false
        self.rules.append((QRegularExpression(r'\b(?:true|false)\b'), bool_format))

        # null
        self.rules.append((QRegularExpression(r'\bnull\b'), null_format))

        # braces / brackets
        self.rules.append((QRegularExpression(r'[\{\}\[\]]'), brace_format))

    def highlightBlock(self, text: str) -> None:
        for pattern, text_format in self.rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                start = match.capturedStart()
                length = match.capturedLength()

                # String values are matched with leading colon/spaces -> color only the string
                if pattern.pattern().startswith(r':\s*"'):
                    full = match.captured(0)
                    quote_index = full.find('"')
                    if quote_index >= 0:
                        start += quote_index
                        length -= quote_index

                self.setFormat(start, length, text_format)


class JsonPreviewDialog(QDialog):
    """
    Read-only dialog to display JSON content in a scrollable and readable way.
    """

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)

        self.file_path = str(file_path)
        self.setWindowTitle(Path(self.file_path).name)
        self.resize(860, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel(Path(self.file_path).name)
        title.setObjectName("PageTitle")
        root.addWidget(title)

        self.path_label = QLabel(self.file_path)
        self.path_label.setObjectName("ParameterPathLabel")
        self.path_label.setWordWrap(True)
        root.addWidget(self.path_label)

        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setProperty("jsonEditor", True)
        self.editor.setLineWrapMode(QTextEdit.NoWrap)
        self.highlighter = JsonSyntaxHighlighter(self.editor.document())
        root.addWidget(self.editor, 1)

        btn_row = QHBoxLayout()

        self.btn_copy_path = QPushButton("Pfad kopieren")
        self.btn_copy_path.clicked.connect(self.copy_path)

        self.btn_copy_content = QPushButton("Inhalt kopieren")
        self.btn_copy_content.clicked.connect(self.copy_content)

        self.btn_close = QPushButton("Schließen")
        self.btn_close.clicked.connect(self.accept)

        btn_row.addWidget(self.btn_copy_path)
        btn_row.addWidget(self.btn_copy_content)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)

        root.addLayout(btn_row)

        self.load_json()

    def load_json(self) -> None:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))

    def copy_path(self) -> None:
        QApplication.clipboard().setText(self.file_path)

    def copy_content(self) -> None:
        QApplication.clipboard().setText(self.editor.toPlainText())