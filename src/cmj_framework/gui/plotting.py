import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.collections as mcoll
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QSizePolicy

# Ensure the project root is importable (dev only; avoid polluting sys.path in PyInstaller bundles)
if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from cmj_framework.data_processing.run_processing_temp_data import TempProcessedData
from cmj_framework.utils.metrics import JumpMetrics
from cmj_framework.utils.visualisation import CMJPlot


class CMJCanvas(FigureCanvas):
    """Matplotlib canvas wrapper for Qt embedding (single-axis overlay)."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(13, 7.5))
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumSize(1250, 720)
        self.resize(1250, 720)

        self.ax = self.fig.add_subplot(111)
        self._apply_default_layout()

    def _apply_default_layout(self) -> None:
        """
        Keep a larger right margin so legends and annotations fit inside the figure.
        """
        self.fig.subplots_adjust(left=0.07, right=0.78, top=0.92, bottom=0.10)

    def reset_axes(self) -> None:
        """Clear axis and restore layout."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._apply_default_layout()


class PlottingMixin:
    """Reusable plotting logic for the CMJ viewer widget."""

    def _populate_curves_list(self) -> None:
        current_label = None
        current_item = self.curves_list.currentItem()
        if current_item:
            current_label = current_item.text()

        self.curves_list.blockSignals(True)
        self.curves_list.clear()

        if not self.plotter or not getattr(self.plotter, "lines", None):
            self.curves_list.blockSignals(False)
            return

        for label, obj in self.plotter.lines.items():
            if isinstance(obj, mcoll.PolyCollection):
                continue

            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            is_visible = bool(obj.get_visible()) if hasattr(obj, "get_visible") else True
            item.setCheckState(Qt.Checked if is_visible else Qt.Unchecked)
            self.curves_list.addItem(item)

        if current_label:
            matches = self.curves_list.findItems(current_label, Qt.MatchExactly)
            if matches:
                self.curves_list.setCurrentItem(matches[0])
            elif self.curves_list.count() > 0:
                self.curves_list.setCurrentRow(0)
        elif self.curves_list.count() > 0:
            self.curves_list.setCurrentRow(0)

        self.curves_list.blockSignals(False)

    def _set_all_curves(self, visible: bool) -> None:
        if not self.plotter:
            return

        self.curves_list.blockSignals(True)
        for i in range(self.curves_list.count()):
            item = self.curves_list.item(i)
            item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
            self.plotter.set_curve_visible(item.text(), visible, redraw=False)
        self.curves_list.blockSignals(False)

        self.canvas.draw_idle()

    def _reset_curve_widths(self) -> None:
        if not self.plotter:
            return

        line_types = getattr(self.plotter, "line_types", {}) or {}
        defaults = getattr(self.plotter, "default_widths", {}) or {}

        for label in list(getattr(self.plotter, "lines", {}).keys()):
            category = line_types.get(label, None)
            default_width = defaults.get(category, 2.0)
            try:
                self.plotter.set_line_width(label, float(default_width), redraw=False)
            except Exception:
                pass

        current_item = self.curves_list.currentItem()
        if current_item:
            self._on_curve_selected(current_item, None)

        self.canvas.draw_idle()

    def _on_curve_toggled(self, item: QListWidgetItem) -> None:
        if not self.plotter:
            return

        label = item.text()
        visible = item.checkState() == Qt.Checked
        self.plotter.set_curve_visible(label, visible, redraw=False)
        self.canvas.draw_idle()

    def _on_curve_selected(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        del previous

        if not self.plotter or not current:
            return

        label = current.text()
        obj = self.plotter.lines.get(label)

        line_width = 2.2
        try:
            line_width = float(obj.get_linewidth())
        except Exception:
            pass

        slider_value = int(round(line_width * 10))
        slider_value = max(5, min(60, slider_value))

        self.width_slider.blockSignals(True)
        self.width_slider.setValue(slider_value)
        self.width_slider.blockSignals(False)
        self.width_value.setText(f"{slider_value / 10:.1f}")

    def _on_width_changed(self, slider_val: int) -> None:
        if not self.plotter:
            return

        item = self.curves_list.currentItem()
        if not item:
            return

        label = item.text()
        line_width = slider_val / 10.0

        self.width_value.setText(f"{line_width:.1f}")
        self.plotter.set_line_width(label, line_width, redraw=False)
        self.canvas.draw_idle()

    def _update_info_card(self, trial_name: str) -> None:
        trial = TempProcessedData.get_trial(trial_name)
        if trial is None:
            self.info_text.setText("—")
            return

        json_path = getattr(trial, "json_path", "") or ""
        sampling_rate = float(getattr(trial, "frame_rate", 0.0) or 0.0)

        try:
            sample_count = int(len(trial.F_total))
        except Exception:
            sample_count = 0

        duration = (sample_count / trial.plate_rate) if trial.plate_rate > 0 else 0.0

        lines = [
            f"Name: {trial_name}",
            f"Datei: {os.path.basename(json_path) if json_path else '—'}",
            (f"Sampling: {sampling_rate:.1f} Hz" if sampling_rate > 0 else "Sampling: —"),
            (f"Dauer: {duration:.2f} s" if duration > 0 else "Dauer: —"),
        ]

        self.info_text.setText("\n".join(lines))

    def _compute_metrics_info(self, trial) -> Dict[str, Any]:
        try:
            jump_metrics = JumpMetrics(
                L=np.asarray(trial.Fz_l, dtype=float),
                R=np.asarray(trial.Fz_r, dtype=float),
                T=np.asarray(trial.F_total, dtype=float),
                X=np.asarray(trial.trajectory, dtype=float),
                rate=float(trial.frame_rate),
            )
            return dict(jump_metrics.all_metrics)
        except Exception:
            return {}

    def _load_trial(self, trial_name: str) -> None:
        trial = TempProcessedData.get_trial(trial_name)
        if trial is None:
            return

        self._update_info_card(trial_name)

        self.canvas.reset_axes()

        self.plotter = CMJPlot(
            L=trial.Fz_l,
            R=trial.Fz_r,
            T=trial.F_total,
            X=trial.trajectory,
            rate=trial.plate_rate,
            strict_roi_markers=True,
        )

        info = self._compute_metrics_info(trial)
        if info:
            self.plotter.set_info(info)

        self.plotter.plot_embedded(self.canvas.ax)

        self.plotter.set_zoom_mode(self.zoom_combo.currentText())
        self.plotter.set_view(self.view_combo.currentText())
        self.plotter.set_roi_source(self.roi_combo.currentText())
        self.plotter.set_info_box_visible(self.info_box_check.isChecked())

        self._populate_curves_list()
        self.canvas.draw_idle()

    def change_zoom(self, text: str) -> None:
        if self.plotter:
            self.plotter.set_zoom_mode(text)
            self.canvas.draw_idle()

    def change_view(self, text: str) -> None:
        if self.plotter:
            self.plotter.set_view(text)
            self._populate_curves_list()
            self.canvas.draw_idle()

    def change_roi(self, text: str) -> None:
        if self.plotter:
            self.plotter.set_roi_source(text)
            self.canvas.draw_idle()

    def toggle_info_box(self, checked: bool) -> None:
        if self.plotter:
            self.plotter.set_info_box_visible(bool(checked))
            self.canvas.draw_idle()