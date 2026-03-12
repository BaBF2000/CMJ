import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

import matplotlib.lines as mlines
import matplotlib.collections as mcoll
import matplotlib.patches as mpatches

from .roi import CMJ_ROI
from .plot_annotations import (
    add_event_line,
    add_phase_arrows,
    apply_grid,
    autoscale_y_visible,
    clear_artists,
    compute_detailed_phase_spans,
    compute_global_phase_spans,
    compute_roi_window_from_spans,
    draw_detailed_roi_shading,
    draw_global_phase_bands,
    safe_span,
    style_axes,
)


@dataclass(frozen=True)
class PlotTheme:
    """
    Light clinical/scientific theme.

    Notes
    -----
    - Keep colors here so QSS and plots stay consistent.
    - Plot remains readable in reports/screenshots.
    """
    bg: str = "#ffffff"
    fg: str = "#1f2937"
    grid: str = "#d1d5db"
    spine: str = "#94a3b8"

    total_force: str = "#111827"
    left_force: str = "#f59e0b"
    right_force: str = "#7c3aed"
    marker_traj: str = "#64748b"

    com_pos: str = "#111827"
    com_vel: str = "#2563eb"
    com_acc: str = "#0ea5e9"

    bodyweight: str = "#fb923c"

    roi_braking: str = "#94a3b8"
    roi_p1: str = "#f59e0b"
    roi_p2: str = "#60a5fa"
    roi_flight: str = "#e2e8f0"
    roi_landing: str = "#a78bfa"

    band_eccentric: str = "#9ca3af"
    band_concentric: str = "#fbbf24"
    band_flight: str = "#d1d5db"
    band_landing: str = "#c4b5fd"


class CMJPlot:
    """
    Interactive CMJ plot (driven by the app).

    Visualization only: scaling/offsets are for display and do not affect metrics computation.
    """

    def __init__(
        self,
        L,
        R,
        T,
        X,
        rate: float,
        roi_L=None,
        roi_R=None,
        roi_T=None,
        com_L=None,
        com_R=None,
        com_T=None,
        theme: Optional[PlotTheme] = None,
        marker_to_cm: bool = True,
        com_to_cm: bool = True,
        visual_align_to_force_baseline: bool = True,
        strict_roi_markers: bool = True,
    ):
        self.theme = theme or PlotTheme()

        # --- Raw signals ---
        self.L_raw = np.asarray(L, dtype=float)
        self.R_raw = np.asarray(R, dtype=float)
        self.T_raw = np.asarray(T, dtype=float)
        self.traj_raw = np.asarray(X, dtype=float)

        self.rate = float(rate)
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("rate must be a positive finite number")

        self.t = np.arange(len(self.T_raw), dtype=float) / self.rate

        # --- ROI computation ---
        self.roi_L = roi_L or CMJ_ROI(self.L_raw, self.traj_raw, self.rate)
        self.roi_R = roi_R or CMJ_ROI(self.R_raw, self.traj_raw, self.rate)
        self.roi_T = roi_T or CMJ_ROI(self.T_raw, self.traj_raw, self.rate)

        self.roi_map = {"total": self.roi_T, "left": self.roi_L, "right": self.roi_R}
        self.active_roi_name = "total"
        self.active_roi = self.roi_T

        # --- COM ---
        self.com_L = com_L or getattr(self.roi_L, "cm_kin", None)
        self.com_R = com_R or getattr(self.roi_R, "cm_kin", None)
        self.com_T = com_T or getattr(self.roi_T, "cm_kin", None)
        self.com = self.com_T

        # --- View management ---
        self.current_view = "all"
        self.y_units = {
            "all": ("Kraft / Kinematik (visuell)", "N + skaliert"),
            "force": ("Kraft", "N"),
            "trajectory": ("Trajektorie", "cm"),
            "com_position": ("Schwerpunktposition", "cm"),
            "velocity": ("Schwerpunktgeschwindigkeit", "cm/s"),
            "acceleration": ("Schwerpunktbeschleunigung", "cm/s²"),
        }

        self.line_types = {
            "Gesamtkraft": "force",
            "Kraft links": "force",
            "Kraft rechts": "force",
            "Körpergewicht": "force",
            "Gemessene Trajektorie": "trajectory",
            "Berechnete Schwerpunkttrajektorie": "com_position",
            "Schwerpunktgeschwindigkeit": "velocity",
            "Schwerpunktbeschleunigung": "acceleration",
            "Bremsphase": "force",
            "Abstoßphase 1": "force",
            "Abstoßphase 2": "force",
            "Flugphase_Detail": "force",
            "Landephase_Detail": "force",
        }

        self.lines: Dict[str, Any] = {}
        self.shaded: Dict[str, Any] = {}
        self.phase_bands: Dict[str, Any] = {}

        self.fig = None
        self.ax = None
        self.zoom_mode = "roi"

        self._legend_curves = None
        self._legend_phases = None
        self._info_artist = None

        self.user_visibility: Dict[str, bool] = {}

        self.default_widths = {
            "force": 2.2,
            "trajectory": 1.8,
            "com_position": 1.6,
            "velocity": 1.6,
            "acceleration": 1.6,
        }

        self._marker_to_cm = bool(marker_to_cm)
        self._com_to_cm = bool(com_to_cm)
        self._align = bool(visual_align_to_force_baseline)
        self._strict_roi_markers = bool(strict_roi_markers)

        self._info: Dict[str, Any] = {}
        self._show_info_box: bool = True

        self._event_artists: list[Any] = []
        self._phase_label_artists: list[Any] = []

        self._prepare_visual_signals()

    # ---------------------------
    # Public API
    # ---------------------------
    def set_info(self, info: Dict[str, Any]) -> None:
        """Attach a dict of scientific info to show on the plot."""
        self._info = dict(info or {})

    def set_info_box_visible(self, visible: bool) -> None:
        """Show or hide the info box drawn inside the plot."""
        self._show_info_box = bool(visible)

        if self.ax is None:
            return

        self.plot_embedded(self.ax)
        self.set_zoom_mode(self.zoom_mode)
        self.set_view(self.current_view, redraw=False)
        self.set_roi_source(self.active_roi_name)

        if self.fig is not None and self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

    # ==========================================================
    # Visual signals
    # ==========================================================
    def _prepare_visual_signals(self) -> None:
        """Baseline-correct force around ROI local minimum; align other signals visually."""
        roi = self.roi_T
        idx0 = int(getattr(roi, "local_min_f", 0))
        idx0 = max(0, min(idx0, len(self.T_raw) - 1))

        self.plot_force = self.T_raw - self.T_raw[idx0]

        self.L_plot = self.L_raw - self.L_raw[0] + self.plot_force[0]
        self.R_plot = self.R_raw - self.R_raw[0] + self.plot_force[0]

        traj = self.traj_raw - self.traj_raw[0]
        if self._marker_to_cm:
            traj = traj * 10
        self.traj_plot = traj + self.plot_force[0]

        self.pos_plot = self.vel_plot = self.acc_plot = None
        if self.com is None:
            return

        pos = getattr(self.com, "position", None)
        vel = getattr(self.com, "velocity", None)
        acc = getattr(self.com, "acceleration", None)

        if pos is None or vel is None or acc is None:
            warnings.warn("COM object missing position/velocity/acceleration; plotting without COM.")
            return

        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        acc = np.asarray(acc, dtype=float)

        if self._com_to_cm:
            pos = pos * 1000.0
            vel = vel * 100.0
            acc = acc * 10.0

        if self._align:
            pos = pos - pos[0] + self.plot_force[0]
            vel = vel - vel[0] + self.plot_force[0]
            acc = acc - acc[0] + self.plot_force[0]

        self.pos_plot = pos
        self.vel_plot = vel
        self.acc_plot = acc

    # ==========================================================
    # Phase / ROI helpers
    # ==========================================================
    def _safe_span(self, span):
        return safe_span(span, len(self.t))

    def _global_phase_spans(self) -> Dict[str, Any]:
        return compute_global_phase_spans(self.active_roi, len(self.t))

    def _detailed_phase_spans(self) -> Dict[str, Any]:
        return compute_detailed_phase_spans(self.active_roi, len(self.t))

    def _roi_window(self):
        spans = self._global_phase_spans()
        return compute_roi_window_from_spans(
            self.t,
            [
                spans.get("stand_to_start"),
                spans.get("eccentric"),
                spans.get("concentric"),
                spans.get("flight"),
                spans.get("landing"),
            ],
        )

    def _apply_xlim_roi(self) -> None:
        if self.ax is None:
            return
        t0, t1 = self._roi_window()
        if t1 > t0:
            self.ax.set_xlim(t0, t1)

    def _event_allowed_in_current_zoom(self, x_s: float) -> bool:
        if not self._strict_roi_markers:
            return True
        if self.zoom_mode != "roi":
            return True
        t0, t1 = self._roi_window()
        return t0 <= x_s <= t1

    # ==========================================================
    # Drawing helpers
    # ==========================================================
    def _style_axes(self) -> None:
        if self.ax is None:
            return
        style_axes(self.ax, self.fig, self.theme)

    def _apply_grid(self) -> None:
        if self.ax is None:
            return
        apply_grid(self.ax, self.theme)

    def _clear_event_markers(self) -> None:
        clear_artists(self._event_artists)

    def _clear_phase_labels(self) -> None:
        clear_artists(self._phase_label_artists)

    def _clear_global_phase_bands(self) -> None:
        for band in list(self.phase_bands.values()):
            try:
                band.remove()
            except Exception:
                pass
        self.phase_bands.clear()

    def _clear_info_box(self) -> None:
        if self._info_artist is not None:
            try:
                self._info_artist.remove()
            except Exception:
                pass
            self._info_artist = None

    def _draw_global_phase_bands(self) -> None:
        if self.ax is None:
            return
        draw_global_phase_bands(
            ax=self.ax,
            t=self.t,
            phase_bands=self.phase_bands,
            spans=self._global_phase_spans(),
            theme=self.theme,
        )

    def _draw_roi_shading(self) -> None:
        if self.ax is None:
            return
        draw_detailed_roi_shading(
            ax=self.ax,
            t=self.t,
            plot_force=self.plot_force,
            shaded=self.shaded,
            spans=self._detailed_phase_spans(),
            theme=self.theme,
        )
        self.lines.update(self.shaded)

    def _draw_event_markers(self) -> None:
        """Add simplified vertical event markers from the active ROI."""
        if self.ax is None:
            return

        self._clear_event_markers()

        roi = self.active_roi
        n = len(self.t)
        if n == 0 or roi is None:
            return

        def _idx_to_time(idx: Any) -> Optional[float]:
            try:
                i = int(idx)
            except Exception:
                return None
            i = max(0, min(i, n - 1))
            return float(self.t[i])

        entries = [
            ("Standphase", _idx_to_time(getattr(roi, "stand", None))),
            ("Bewegungsbeginn", _idx_to_time(getattr(roi, "start", None))),
            ("Absprung", _idx_to_time(getattr(roi, "takeoff_idx", None))),
        ]

        y = 0.78
        step = 0.07
        for label, x_s in entries:
            if x_s is None:
                continue
            if not self._event_allowed_in_current_zoom(x_s):
                continue
            add_event_line(
                ax=self.ax,
                x_s=x_s,
                text=label,
                y_pos_ax=y,
                theme=self.theme,
                artists_store=self._event_artists,
            )
            y -= step

    def _draw_phase_labels(self) -> None:
        """Draw phase arrows below the curves."""
        if self.ax is None:
            return

        self._clear_phase_labels()
        add_phase_arrows(
            ax=self.ax,
            t=self.t,
            spans=self._global_phase_spans(),
            theme=self.theme,
            artists_store=self._phase_label_artists,
            is_allowed_cb=self._event_allowed_in_current_zoom,
        )

    def _autoscale_y_visible(self) -> None:
        if self.ax is None:
            return
        autoscale_y_visible(self.ax, self.lines)

    # ==========================================================
    # Legend / info box
    # ==========================================================
    def _place_legend(self) -> None:
        if self.fig is None or self.ax is None:
            return

        try:
            self.fig.subplots_adjust(left=0.07, right=0.74, top=0.92, bottom=0.10)
        except Exception:
            pass

        curve_handles = []
        curve_labels = []

        for label, obj in self.lines.items():
            if not getattr(obj, "get_visible", lambda: True)():
                continue
            if isinstance(obj, mlines.Line2D):
                curve_handles.append(obj)
                curve_labels.append(label)

        phase_handles = []
        phase_labels = []

        if self.current_view in ("all", "force"):
            visible_band_entries = [
                ("Exzentrische Phase", self.theme.band_eccentric),
                ("Konzentrische Phase", self.theme.band_concentric),
                ("Flugphase", self.theme.band_flight),
                ("Landephase", self.theme.band_landing),
            ]

            for label, color in visible_band_entries:
                band_obj = self.phase_bands.get(label)
                if band_obj is not None and getattr(band_obj, "get_visible", lambda: True)():
                    phase_handles.append(mpatches.Patch(color=color, alpha=0.15))
                    phase_labels.append(label)

            detailed_entries = [
                ("Bremsphase", self.theme.roi_braking, "Bremsphase"),
                ("Abstoßphase 1", self.theme.roi_p1, "Abstoßphase 1"),
                ("Abstoßphase 2", self.theme.roi_p2, "Abstoßphase 2"),
                ("Flugphase_Detail", self.theme.roi_flight, "Flugphase (ROI)"),
                ("Landephase_Detail", self.theme.roi_landing, "Landephase (ROI)"),
            ]

            for internal_label, color, display_label in detailed_entries:
                obj = self.shaded.get(internal_label)
                if obj is not None and getattr(obj, "get_visible", lambda: True)():
                    phase_handles.append(mpatches.Patch(color=color, alpha=0.25))
                    phase_labels.append(display_label)

        for attr_name in ("_legend_curves", "_legend_phases"):
            old_legend = getattr(self, attr_name, None)
            if old_legend is not None:
                try:
                    old_legend.remove()
                except Exception:
                    pass
                setattr(self, attr_name, None)

        if curve_handles:
            self._legend_curves = self.ax.legend(
                curve_handles,
                curve_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.00),
                borderaxespad=0.0,
                frameon=True,
                title="Kurven",
                fontsize=9,
                title_fontsize=10,
            )

            try:
                frame = self._legend_curves.get_frame()
                frame.set_facecolor("#ffffff")
                frame.set_edgecolor(self.theme.spine)

                title = self._legend_curves.get_title()
                title.set_color(self.theme.fg)

                for text in self._legend_curves.get_texts():
                    text.set_color(self.theme.fg)

                self.ax.add_artist(self._legend_curves)
            except Exception:
                pass

        if phase_handles:
            y_anchor = 0.50 if curve_handles else 1.00

            self._legend_phases = self.ax.legend(
                phase_handles,
                phase_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, y_anchor),
                borderaxespad=0.0,
                frameon=True,
                title="Phasen",
                fontsize=9,
                title_fontsize=10,
            )

            try:
                frame = self._legend_phases.get_frame()
                frame.set_facecolor("#ffffff")
                frame.set_edgecolor(self.theme.spine)

                title = self._legend_phases.get_title()
                title.set_color(self.theme.fg)

                for text in self._legend_phases.get_texts():
                    text.set_color(self.theme.fg)
            except Exception:
                pass

    def _draw_info_box(self) -> None:
        if self.ax is None:
            return

        self._clear_info_box()

        if not self._show_info_box:
            return

        roi = self.active_roi
        lines = []

        try:
            lines.append(f"ROI-Quelle: {self.active_roi_name}")
            lines.append(f"KG: {float(roi.bodyweight):.1f} N   Masse: {float(roi.mass):.2f} kg")
            lines.append(f"Sprunghöhe: {float(roi.jump_height):.3f} m   Flugzeit: {float(roi.flight_time):.3f} s")
        except Exception:
            pass

        extra = self._info or {}

        def _get(path: str):
            cur: Any = extra
            for p in path.split("."):
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur

        rsi = _get("RSI_modified")
        if rsi is not None:
            lines.append(f"RSI_mod: {float(rsi):.3f}")

        tt = _get("PhaseDurations.Time_to_takeoff")
        if tt is not None:
            lines.append(f"Zeit bis Absprung: {float(tt):.3f} s")

        pk = _get("PeakForce.propulsion")
        if pk is not None:
            lines.append(f"Max. Abstoßkraft: {float(pk):.1f} N")

        if not lines:
            return

        txt = "\n".join(lines)
        self._info_artist = self.ax.text(
            0.01,
            0.99,
            txt,
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=self.theme.fg,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#ffffff",
                edgecolor=self.theme.spine,
                alpha=0.92,
            ),
        )

    # ==========================================================
    # View selection / ROI source
    # ==========================================================
    def set_zoom_mode(self, mode: str) -> None:
        if mode not in ("roi", "full"):
            raise ValueError("zoom mode must be 'roi' or 'full'")
        self.zoom_mode = mode

        if self.ax is None:
            return

        if mode == "roi":
            self._apply_xlim_roi()
        else:
            self.ax.set_xlim(float(self.t[0]), float(self.t[-1]))

        self._draw_event_markers()
        self._draw_phase_labels()
        self._autoscale_y_visible()
        self._place_legend()

        if self.fig is not None and self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

    def set_view(self, mode: str, redraw: bool = True) -> None:
        if self.ax is None:
            raise RuntimeError("Plot not created yet. Call plot_embedded(ax) first.")
        if mode not in self.y_units:
            raise ValueError(f"Invalid view mode: {mode}")

        self.current_view = mode
        self._apply_visibility_rules()

        if redraw and self.fig is not None and self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

    def set_roi_source(self, source: str) -> None:
        if self.ax is None:
            raise RuntimeError("Plot not created yet. Call plot_embedded(ax) first.")
        if source not in self.roi_map:
            raise ValueError("ROI source must be 'total', 'left' or 'right'")

        self.active_roi_name = source
        self.active_roi = self.roi_map[source]

        self._draw_global_phase_bands()
        self._draw_roi_shading()

        if self.zoom_mode == "roi":
            self._apply_xlim_roi()

        self._draw_event_markers()
        self._draw_phase_labels()
        self._draw_info_box()
        self._apply_grid()
        self._autoscale_y_visible()
        self._place_legend()

        if self.fig is not None and self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

    def _apply_visibility_rules(self) -> None:
        if self.ax is None:
            return

        mode = self.current_view

        for label, obj in self.lines.items():
            if isinstance(obj, mcoll.PolyCollection) and label in (
                "Bremsphase",
                "Abstoßphase 1",
                "Abstoßphase 2",
                "Flugphase_Detail",
                "Landephase_Detail",
            ):
                base_visible = mode in ("all", "force")
            else:
                signal_type = self.line_types.get(label, None)
                base_visible = True if mode == "all" else (signal_type == mode)

            if label in self.user_visibility:
                base_visible = base_visible and bool(self.user_visibility[label])

            try:
                obj.set_visible(base_visible)
            except Exception:
                pass

        show_bands = mode in ("all", "force")
        for band in self.phase_bands.values():
            try:
                band.set_visible(show_bands)
            except Exception:
                pass

        label_txt, unit = self.y_units[mode]
        self.ax.set_ylabel(f"{label_txt} [{unit}]")
        self._apply_grid()
        self._autoscale_y_visible()
        self._place_legend()

    def set_curve_visible(self, label: str, visible: bool, redraw: bool = True) -> None:
        self.user_visibility[label] = bool(visible)
        self._apply_visibility_rules()
        self._place_legend()
        if redraw and self.fig and self.fig.canvas:
            self.fig.canvas.draw_idle()

    def set_line_width(self, label: str, width: float, redraw: bool = True) -> None:
        obj = self.lines.get(label)
        if obj is None:
            return
        if isinstance(obj, mlines.Line2D):
            obj.set_linewidth(float(width))
        if redraw and self.fig and self.fig.canvas:
            self.fig.canvas.draw_idle()

    # ==========================================================
    # Plotting
    # ==========================================================
    def plot_embedded(self, ax) -> None:
        self.ax = ax
        self.fig = ax.figure

        self.ax.clear()
        self.lines.clear()
        self.shaded.clear()
        self._clear_global_phase_bands()
        self._clear_event_markers()
        self._clear_phase_labels()
        self._clear_info_box()

        self._style_axes()
        self._draw_global_phase_bands()

        (self.lines["Gesamtkraft"],) = self.ax.plot(
            self.t, self.plot_force, color=self.theme.total_force, label="Gesamtkraft"
        )
        self.lines["Gesamtkraft"].set_linewidth(self.default_widths["force"])

        (self.lines["Kraft links"],) = self.ax.plot(
            self.t, self.L_plot, "--", color=self.theme.left_force, label="Kraft links"
        )
        self.lines["Kraft links"].set_linewidth(self.default_widths["force"])

        (self.lines["Kraft rechts"],) = self.ax.plot(
            self.t, self.R_plot, "--", color=self.theme.right_force, label="Kraft rechts"
        )
        self.lines["Kraft rechts"].set_linewidth(self.default_widths["force"])

        (self.lines["Gemessene Trajektorie"],) = self.ax.plot(
            self.t, self.traj_plot, color=self.theme.marker_traj, label="Gemessene Trajektorie"
        )
        self.lines["Gemessene Trajektorie"].set_linewidth(self.default_widths["trajectory"])

        if self.pos_plot is not None:
            (self.lines["Berechnete Schwerpunkttrajektorie"],) = self.ax.plot(
                self.t,
                self.pos_plot,
                "--",
                color=self.theme.com_pos,
                label="Berechnete Schwerpunkttrajektorie",
            )
            self.lines["Berechnete Schwerpunkttrajektorie"].set_linewidth(self.default_widths["com_position"])

            (self.lines["Schwerpunktgeschwindigkeit"],) = self.ax.plot(
                self.t, self.vel_plot, color=self.theme.com_vel, label="Schwerpunktgeschwindigkeit"
            )
            self.lines["Schwerpunktgeschwindigkeit"].set_linewidth(self.default_widths["velocity"])

            (self.lines["Schwerpunktbeschleunigung"],) = self.ax.plot(
                self.t, self.acc_plot, color=self.theme.com_acc, label="Schwerpunktbeschleunigung"
            )
            self.lines["Schwerpunktbeschleunigung"].set_linewidth(self.default_widths["acceleration"])

        roi = self.active_roi
        idx0 = int(getattr(roi, "local_min_f", 0))
        idx0 = max(0, min(idx0, len(self.T_raw) - 1))
        bw = float(getattr(roi, "bodyweight", 0.0)) - self.T_raw[idx0]
        self.lines["Körpergewicht"] = self.ax.axhline(
            bw, linestyle=":", color=self.theme.bodyweight, label="Körpergewicht"
        )

        self._draw_roi_shading()

        self.ax.set_xlabel("Zeit [s]")
        label, unit = self.y_units[self.current_view]
        self.ax.set_ylabel(f"{label} [{unit}]")
        self.ax.set_title("Countermovement Jump (CMJ) – Bodenreaktionskraft, Phasen und Schwerpunktskinematik")

        self._draw_info_box()
        self._apply_grid()

        if self.zoom_mode == "roi":
            self._apply_xlim_roi()
        else:
            self.ax.set_xlim(float(self.t[0]), float(self.t[-1]))

        self.set_view(self.current_view, redraw=False)

        self._draw_event_markers()
        self._draw_phase_labels()
        self._place_legend()