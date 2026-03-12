import sys
from pathlib import Path

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QFrame,
    QPushButton,
    QCheckBox,
    QListWidget,
    QSlider,
    QGroupBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QTimer

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

# Ensure the project root is importable (dev only; avoid polluting sys.path in PyInstaller bundles)
if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from cmj_framework.data_processing.run_processing_temp_data import TempProcessedData
from cmj_framework.gui.plotting import CMJCanvas, PlottingMixin
from cmj_framework.gui.trial_review import TrialReviewMixin


class CMJViewerWidget(QWidget, PlottingMixin, TrialReviewMixin):
    """Embedded CMJ viewer widget."""

    trialRejected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("CMJViewer")

        self.plotter = None
        self.decisions: dict[str, str] = {}
        self._current_trial: str | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ===== Left controls =====
        controls = QWidget()
        controls.setObjectName("CMJViewerControls")
        controls.setFixedWidth(300)

        vbox = QVBoxLayout(controls)
        vbox.setAlignment(Qt.AlignTop)
        vbox.setSpacing(10)
        vbox.setContentsMargins(10, 10, 10, 10)

        self.status_label = QLabel("⏳ Entscheidung ausstehend")
        self.status_label.setObjectName("CMJStatusBadge")
        self.status_label.setProperty("state", "pending")
        self.status_label.setWordWrap(True)
        vbox.addWidget(self.status_label)

        vbox.addWidget(QLabel("Trial:"))
        self.trial_combo = QComboBox()
        self.trial_combo.setObjectName("CMJTrialCombo")
        self.trial_combo.addItems(TempProcessedData.list_trials())
        self.trial_combo.currentTextChanged.connect(self.change_trial)
        vbox.addWidget(self.trial_combo)

        # ===== Trial info card =====
        self.info_card = QFrame()
        self.info_card.setObjectName("CMJInfoCard")
        info_layout = QVBoxLayout(self.info_card)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(6)

        self.info_title = QLabel("Trial-Info")
        self.info_title.setObjectName("CMJInfoTitle")
        info_layout.addWidget(self.info_title)

        self.info_text = QLabel("—")
        self.info_text.setObjectName("CMJInfoText")
        self.info_text.setWordWrap(True)
        info_layout.addWidget(self.info_text)

        vbox.addWidget(self.info_card)

        # ===== Review block =====
        self.review_box = QFrame()
        self.review_box.setObjectName("CMJReviewCard")

        review_layout = QVBoxLayout(self.review_box)
        review_layout.setContentsMargins(10, 10, 10, 10)
        review_layout.setSpacing(8)

        self.review_text = QLabel("Möchten Sie diesen Versuch behalten?")
        self.review_text.setObjectName("CMJReviewText")
        self.review_text.setWordWrap(True)
        review_layout.addWidget(self.review_text)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.btn_yes = QPushButton("Ja")
        self.btn_yes.setObjectName("Primary")

        self.btn_no = QPushButton("Nein")
        self.btn_no.setObjectName("Danger")

        self.btn_later = QPushButton("Später")
        self.btn_later.setObjectName("Secondary")

        button_row.addWidget(self.btn_yes)
        button_row.addWidget(self.btn_no)
        button_row.addWidget(self.btn_later)
        review_layout.addLayout(button_row)

        vbox.addWidget(self.review_box)

        self.btn_yes.clicked.connect(self._keep_current_trial)
        self.btn_no.clicked.connect(self._reject_current_trial)
        self.btn_later.clicked.connect(self._later_current_trial)

        vbox.addWidget(QLabel("Zoom:"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.setObjectName("CMJZoomCombo")
        self.zoom_combo.addItems(["roi", "full"])
        self.zoom_combo.currentTextChanged.connect(self.change_zoom)
        vbox.addWidget(self.zoom_combo)

        vbox.addWidget(QLabel("Ansicht:"))
        self.view_combo = QComboBox()
        self.view_combo.setObjectName("CMJViewCombo")
        self.view_combo.addItems(["all", "force", "trajectory", "com_position", "velocity", "acceleration"])
        self.view_combo.currentTextChanged.connect(self.change_view)
        vbox.addWidget(self.view_combo)

        self.info_box_check = QCheckBox("Info-Box anzeigen")
        self.info_box_check.setObjectName("CMJInfoBoxToggle")
        self.info_box_check.setChecked(True)
        self.info_box_check.toggled.connect(self.toggle_info_box)
        vbox.addWidget(self.info_box_check)

        # ===== Curves =====
        curves_box = QGroupBox("Kurven")
        curves_box.setObjectName("CMJCurvesBox")
        curves_layout = QVBoxLayout(curves_box)
        curves_layout.setSpacing(6)

        self.curves_list = QListWidget()
        self.curves_list.setObjectName("CMJCurvesList")
        self.curves_list.setMinimumHeight(170)
        self.curves_list.itemChanged.connect(self._on_curve_toggled)
        self.curves_list.currentItemChanged.connect(self._on_curve_selected)
        curves_layout.addWidget(self.curves_list)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_curves_all = QPushButton("Alle")
        self.btn_curves_all.setObjectName("Secondary")

        self.btn_curves_none = QPushButton("Keine")
        self.btn_curves_none.setObjectName("Secondary")

        self.btn_curves_reset = QPushButton("Reset")
        self.btn_curves_reset.setObjectName("Secondary")

        action_row.addWidget(self.btn_curves_all)
        action_row.addWidget(self.btn_curves_none)
        action_row.addWidget(self.btn_curves_reset)
        curves_layout.addLayout(action_row)

        self.btn_curves_all.clicked.connect(lambda: self._set_all_curves(True))
        self.btn_curves_none.clicked.connect(lambda: self._set_all_curves(False))
        self.btn_curves_reset.clicked.connect(self._reset_curve_widths)

        width_row = QHBoxLayout()
        width_row.setSpacing(8)
        width_row.addWidget(QLabel("Linienstärke:"))

        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setObjectName("CMJLineWidthSlider")
        self.width_slider.setRange(5, 60)
        self.width_slider.setValue(22)
        self.width_slider.valueChanged.connect(self._on_width_changed)
        width_row.addWidget(self.width_slider, 1)

        self.width_value = QLabel("2.2")
        self.width_value.setObjectName("CMJLineWidthValue")
        self.width_value.setFixedWidth(44)
        width_row.addWidget(self.width_value)

        curves_layout.addLayout(width_row)
        vbox.addWidget(curves_box)

        vbox.addWidget(QLabel("ROI-Quelle:"))
        self.roi_combo = QComboBox()
        self.roi_combo.setObjectName("CMJROICombo")
        self.roi_combo.addItems(["total", "left", "right"])
        self.roi_combo.currentTextChanged.connect(self.change_roi)
        vbox.addWidget(self.roi_combo)

        vbox.addStretch(1)

        # ===== Right plot area =====
        plot_container = QWidget()
        plot_container.setObjectName("CMJPlotAreaContainer")
        plot_container_layout = QVBoxLayout(plot_container)
        plot_container_layout.setContentsMargins(0, 0, 0, 0)
        plot_container_layout.setSpacing(0)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setObjectName("CMJPlotScrollArea")
        self.plot_scroll.setWidgetResizable(False)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        plot_area = QWidget()
        plot_area.setObjectName("CMJPlotArea")
        plot_area.setMinimumSize(1280, 760)

        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        plot_layout.setSpacing(8)

        self.canvas = CMJCanvas(parent=plot_area)
        self.canvas.setMinimumSize(1250, 720)

        self.toolbar = NavigationToolbar2QT(self.canvas, plot_area)
        self.toolbar.setObjectName("CMJMatplotlibToolbar")

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)

        self.plot_scroll.setWidget(plot_area)
        plot_container_layout.addWidget(self.plot_scroll)

        root.addWidget(controls)
        root.addWidget(plot_container, 1)

        trials = TempProcessedData.list_trials()
        if trials:
            first_trial = trials[0]

            self.trial_combo.blockSignals(True)
            self.trial_combo.setCurrentText(first_trial)
            self.trial_combo.blockSignals(False)

            self._current_trial = first_trial
            self._load_trial(first_trial)
            self._show_review_panel(first_trial)

    def refresh_trials(self) -> None:
        current = self.trial_combo.currentText()
        trials = TempProcessedData.list_trials()

        self.trial_combo.blockSignals(True)
        self.trial_combo.clear()
        self.trial_combo.addItems(trials)

        if current and current in trials:
            self.trial_combo.setCurrentText(current)
        elif trials:
            self.trial_combo.setCurrentText(trials[0])

        self.trial_combo.blockSignals(False)

        selected = self.trial_combo.currentText()
        if selected:
            self._current_trial = selected
            self._load_trial(selected)
            self._show_review_panel(selected)

    def select_trial(self, trial_name: str) -> None:
        names = TempProcessedData.list_trials()
        if trial_name not in names:
            return

        if self._current_trial == trial_name:
            return

        self.trial_combo.blockSignals(True)
        self.trial_combo.setCurrentText(trial_name)
        self.trial_combo.blockSignals(False)

        self._current_trial = trial_name
        self._load_trial(trial_name)
        self._show_review_panel(trial_name)

    def change_trial(self, name: str) -> None:
        if not name:
            return

        if self._current_trial == name:
            return

        self._current_trial = name
        self._load_trial(name)
        QTimer.singleShot(0, lambda: self._show_review_panel(name))