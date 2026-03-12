from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QLabel, QSlider

from cmj_framework.gui import plotting



class FakeCanvas:
    def __init__(self):
        self.draw_calls = 0
        self.reset_calls = 0
        self.ax = object()

    def draw_idle(self):
        self.draw_calls += 1

    def reset_axes(self):
        self.reset_calls += 1


class FakeLine:
    def __init__(self, visible=True, linewidth=2.2):
        self._visible = visible
        self._linewidth = linewidth

    def get_visible(self):
        return self._visible

    def get_linewidth(self):
        return self._linewidth


class FakePlotter:
    def __init__(self):
        self.lines = {
            "Gesamtkraft": FakeLine(True, 2.2),
            "Kraft links": FakeLine(False, 1.5),
        }
        self.line_types = {
            "Gesamtkraft": "force",
            "Kraft links": "force",
        }
        self.default_widths = {
            "force": 2.2,
        }
        self.visible_calls = []
        self.width_calls = []
        self.zoom_calls = []
        self.view_calls = []
        self.roi_calls = []
        self.info_box_calls = []
        self.info_calls = []
        self.plot_embedded_calls = 0

    def set_curve_visible(self, label, visible, redraw=False):
        self.visible_calls.append((label, visible, redraw))

    def set_line_width(self, label, width, redraw=False):
        self.width_calls.append((label, width, redraw))

    def set_zoom_mode(self, text):
        self.zoom_calls.append(text)

    def set_view(self, text):
        self.view_calls.append(text)

    def set_roi_source(self, text):
        self.roi_calls.append(text)

    def set_info_box_visible(self, checked):
        self.info_box_calls.append(bool(checked))

    def set_info(self, info):
        self.info_calls.append(info)

    def plot_embedded(self, ax):
        self.plot_embedded_calls += 1


class FakeCMJPlot(FakePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init_args = args
        self.init_kwargs = kwargs


class FakeJumpMetrics:
    def __init__(self, *args, **kwargs):
        self.all_metrics = {
            "jump_height": 25.0,
            "RSI_modified": 1.2,
        }


class FakeTrial:
    def __init__(self):
        self.json_path = "/tmp/trial.json"
        self.frame_rate = 200.0
        self.plate_rate = 1000.0
        self.Fz_l = np.array([1.0, 2.0, 3.0])
        self.Fz_r = np.array([1.0, 2.0, 3.0])
        self.F_total = np.array([2.0, 4.0, 6.0])
        self.trajectory = np.array([10.0, 11.0, 12.0])


class FakePlotHost(plotting.PlottingMixin):
    def __init__(self):
        self.curves_list = QListWidget()
        self.canvas = FakeCanvas()
        self.plotter = None
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(5, 60)
        self.width_value = QLabel("")
        self.info_text = QLabel("")
        self.zoom_combo = SimpleNamespace(currentText=lambda: "roi")
        self.view_combo = SimpleNamespace(currentText=lambda: "all")
        self.roi_combo = SimpleNamespace(currentText=lambda: "total")
        self.info_box_check = SimpleNamespace(isChecked=lambda: True)


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_populate_curves_list_adds_visible_and_hidden_lines(qapp):
    """
    Test that _populate_curves_list fills the list with plot lines.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()

    host._populate_curves_list()

    assert host.curves_list.count() == 2
    assert host.curves_list.item(0).text() == "Gesamtkraft"
    assert host.curves_list.item(0).checkState() == Qt.Checked
    assert host.curves_list.item(1).text() == "Kraft links"
    assert host.curves_list.item(1).checkState() == Qt.Unchecked


def test_set_all_curves_updates_all_items_and_plotter(qapp):
    """
    Test that _set_all_curves updates all list items and forwards visibility to the plotter.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()
    host._populate_curves_list()

    host._set_all_curves(True)

    assert host.curves_list.item(0).checkState() == Qt.Checked
    assert host.curves_list.item(1).checkState() == Qt.Checked
    assert ("Gesamtkraft", True, False) in host.plotter.visible_calls
    assert ("Kraft links", True, False) in host.plotter.visible_calls
    assert host.canvas.draw_calls == 1


def test_reset_curve_widths_restores_default_widths(qapp):
    """
    Test that _reset_curve_widths resets curve widths to category defaults.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()
    host._populate_curves_list()

    host._reset_curve_widths()

    assert ("Gesamtkraft", 2.2, False) in host.plotter.width_calls
    assert ("Kraft links", 2.2, False) in host.plotter.width_calls
    assert host.canvas.draw_calls == 1


def test_on_curve_toggled_updates_plotter_visibility(qapp):
    """
    Test that _on_curve_toggled forwards the selected visibility state.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()

    item = QListWidgetItem("Gesamtkraft")
    item.setCheckState(Qt.Unchecked)

    host._on_curve_toggled(item)

    assert ("Gesamtkraft", False, False) in host.plotter.visible_calls
    assert host.canvas.draw_calls == 1


def test_on_curve_selected_updates_slider_and_label(qapp):
    """
    Test that _on_curve_selected updates the width slider and label from the selected line.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()

    item = QListWidgetItem("Gesamtkraft")

    host._on_curve_selected(item, None)

    assert host.width_slider.value() == 22
    assert host.width_value.text() == "2.2"


def test_on_width_changed_updates_plotter_and_label(qapp):
    """
    Test that _on_width_changed updates the selected curve width.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()
    host._populate_curves_list()
    host.curves_list.setCurrentRow(0)

    host._on_width_changed(30)

    assert host.width_value.text() == "3.0"
    assert ("Gesamtkraft", 3.0, False) in host.plotter.width_calls
    assert host.canvas.draw_calls == 1


def test_update_info_card_sets_placeholder_when_trial_missing(monkeypatch, qapp):
    """
    Test that _update_info_card shows a placeholder when the trial is missing.
    """
    host = FakePlotHost()

    monkeypatch.setattr(plotting.TempProcessedData, "get_trial", lambda trial_name: None)

    host._update_info_card("Trial_01")

    assert host.info_text.text() == "—"


def test_update_info_card_displays_trial_information(monkeypatch, qapp):
    """
    Test that _update_info_card displays the expected trial information.
    """
    host = FakePlotHost()
    trial = FakeTrial()

    monkeypatch.setattr(plotting.TempProcessedData, "get_trial", lambda trial_name: trial)

    host._update_info_card("Trial_01")

    text = host.info_text.text()
    assert "Name: Trial_01" in text
    assert "Datei: trial.json" in text
    assert "Sampling: 200.0 Hz" in text
    assert "Dauer:" in text


def test_compute_metrics_info_returns_metrics_dict(monkeypatch, qapp):
    """
    Test that _compute_metrics_info returns the metrics payload.
    """
    host = FakePlotHost()

    monkeypatch.setattr(plotting, "JumpMetrics", FakeJumpMetrics)

    result = host._compute_metrics_info(FakeTrial())

    assert result == {
        "jump_height": 25.0,
        "RSI_modified": 1.2,
    }


def test_compute_metrics_info_returns_empty_dict_on_error(monkeypatch, qapp):
    """
    Test that _compute_metrics_info returns an empty dict if metrics computation fails.
    """
    host = FakePlotHost()

    def raise_metrics(*args, **kwargs):
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(plotting, "JumpMetrics", raise_metrics)

    result = host._compute_metrics_info(FakeTrial())

    assert result == {}


def test_load_trial_builds_plotter_and_updates_ui(monkeypatch, qapp):
    """
    Test that _load_trial loads the trial, creates the plotter, and updates the canvas.
    """
    host = FakePlotHost()
    trial = FakeTrial()

    monkeypatch.setattr(plotting.TempProcessedData, "get_trial", lambda trial_name: trial)
    monkeypatch.setattr(plotting, "CMJPlot", FakeCMJPlot)
    monkeypatch.setattr(host, "_compute_metrics_info", lambda trial: {"jump_height": 20.0})

    host._load_trial("Trial_01")

    assert host.canvas.reset_calls == 1
    assert isinstance(host.plotter, FakeCMJPlot)
    assert host.plotter.info_calls == [{"jump_height": 20.0}]
    assert host.plotter.plot_embedded_calls == 1
    assert host.plotter.zoom_calls == ["roi"]
    assert host.plotter.view_calls == ["all"]
    assert host.plotter.roi_calls == ["total"]
    assert host.plotter.info_box_calls == [True]
    assert host.canvas.draw_calls == 1
    assert host.curves_list.count() == 2


def test_change_zoom_forwards_to_plotter(qapp):
    """
    Test that change_zoom forwards the new zoom mode to the plotter.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()

    host.change_zoom("full")

    assert host.plotter.zoom_calls == ["full"]
    assert host.canvas.draw_calls == 1


def test_change_view_forwards_to_plotter_and_refreshes_curves(monkeypatch, qapp):
    """
    Test that change_view forwards the view and refreshes the curves list.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()

    called = {"populate": 0}

    def fake_populate():
        called["populate"] += 1

    monkeypatch.setattr(host, "_populate_curves_list", fake_populate)

    host.change_view("force")

    assert host.plotter.view_calls == ["force"]
    assert called["populate"] == 1
    assert host.canvas.draw_calls == 1


def test_change_roi_forwards_to_plotter(qapp):
    """
    Test that change_roi forwards the ROI source to the plotter.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()

    host.change_roi("left")

    assert host.plotter.roi_calls == ["left"]
    assert host.canvas.draw_calls == 1


def test_toggle_info_box_forwards_to_plotter(qapp):
    """
    Test that toggle_info_box forwards the visibility state to the plotter.
    """
    host = FakePlotHost()
    host.plotter = FakePlotter()

    host.toggle_info_box(False)

    assert host.plotter.info_box_calls == [False]
    assert host.canvas.draw_calls == 1