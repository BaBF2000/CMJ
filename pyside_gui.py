import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from cmj_word_report import Report

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Berichtgenerator")
        self.setGeometry(300, 300, 600, 120)

        # Widgets
        self.label_datei = QLabel("Datei:")
        self.eingabe_datei = QLineEdit()
        self.btn_durchsuchen = QPushButton("Durchsuchen")
        self.btn_generieren = QPushButton("Bericht erstellen")
        self.label_status = QLabel("")

        # Layouts
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label_datei)
        h_layout.addWidget(self.eingabe_datei)
        h_layout.addWidget(self.btn_durchsuchen)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.btn_generieren)
        v_layout.addWidget(self.label_status)

        self.setLayout(v_layout)

        # Signal connections
        self.btn_durchsuchen.clicked.connect(self.select_file)
        self.btn_generieren.clicked.connect(self.start_processing)

    def select_file(self):
        pfad, _ = QFileDialog.getOpenFileName(self, "Datei auswählen", "", "Alle Dateien (*.*)")
        if pfad:
            self.eingabe_datei.setText(pfad)

    def start_processing(self):
        pfad = self.eingabe_datei.text()
        if pfad:
            try:
                Report(pfad)
                self.label_status.setText("Bericht erfolgreich erstellt!")
            except Exception as e:
                self.label_status.setText(f"Fehler: {e}")
        else:
            self.label_status.setText("Bitte wählen Sie eine Datei aus.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    fenster = Window()
    fenster.show()
    sys.exit(app.exec())

