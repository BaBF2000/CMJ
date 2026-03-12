from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.lines as mlines
import matplotlib.collections as mcoll
import matplotlib.patches as mpatches


def clear_artists(artists: List[Any]) -> None:
    """Safely remove matplotlib artists and clear the list."""
    for artist in artists:
        try:
            artist.remove()
        except Exception:
            pass
    artists.clear()


def clear_artist_dict(artist_dict: Dict[str, Any]) -> None:
    """Safely remove matplotlib artists stored in a dict and clear it."""
    for artist in list(artist_dict.values()):
        try:
            artist.remove()
        except Exception:
            pass
    artist_dict.clear()


def safe_span(span, n_samples: int) -> Optional[Tuple[int, int]]:
    """
    Validate and clamp a span to signal bounds.

    Parameters
    ----------
    span : tuple | None
        Candidate span (start, end).
    n_samples : int
        Signal length.

    Returns
    -------
    tuple(int, int) | None
        Clamped span or None if invalid.
    """
    if not span or len(span) != 2:
        return None

    s, e = span
    if s is None or e is None:
        return None

    s = int(max(0, min(int(s), n_samples - 1)))
    e = int(max(0, min(int(e), n_samples)))

    if e <= s:
        return None
    return s, e


def compute_global_phase_spans(roi, n_samples: int) -> Dict[str, Optional[Tuple[int, int]]]:
    """
    Compute global CMJ phase spans.

    Returns
    -------
    dict
        Keys:
        - eccentric
        - concentric
        - flight
        - landing
        - stand_to_start
    """
    if roi is None:
        return {
            "eccentric": None,
            "concentric": None,
            "flight": None,
            "landing": None,
            "stand_to_start": None,
        }

    stand_idx = getattr(roi, "stand", None)
    start_idx = getattr(roi, "start", None)
    local_min_t = getattr(roi, "local_min_t", None)
    takeoff_idx = getattr(roi, "takeoff_idx", None)
    landing_idx = getattr(roi, "landing_idx", None)
    landing_phase = getattr(roi, "landing_phase", None)

    stand_to_start = None
    if stand_idx is not None and start_idx is not None:
        stand_to_start = safe_span((int(stand_idx), int(start_idx)), n_samples)

    eccentric = None
    if start_idx is not None and local_min_t is not None:
        eccentric = safe_span((int(start_idx), int(local_min_t)), n_samples)

    concentric = None
    if local_min_t is not None and takeoff_idx is not None:
        concentric = safe_span((int(local_min_t), int(takeoff_idx)), n_samples)

    flight = None
    if takeoff_idx is not None and landing_idx is not None:
        flight = safe_span((int(takeoff_idx), int(landing_idx)), n_samples)

    landing = safe_span(landing_phase, n_samples)

    return {
        "eccentric": eccentric,
        "concentric": concentric,
        "flight": flight,
        "landing": landing,
        "stand_to_start": stand_to_start,
    }


def compute_detailed_phase_spans(roi, n_samples: int) -> Dict[str, Optional[Tuple[int, int]]]:
    """
    Compute detailed CMJ phase spans.

    Returns
    -------
    dict
        Keys:
        - braking
        - p1
        - p2
        - flight
        - landing
    """
    if roi is None:
        return {
            "braking": None,
            "p1": None,
            "p2": None,
            "flight": None,
            "landing": None,
        }

    braking = safe_span(getattr(roi, "eccentric_phases", {}).get("braking"), n_samples)
    p1 = safe_span(getattr(roi, "concentric_phases", {}).get("propulsion_p1"), n_samples)
    p2 = safe_span(getattr(roi, "concentric_phases", {}).get("propulsion_p2"), n_samples)
    landing = safe_span(getattr(roi, "landing_phase", None), n_samples)

    takeoff_idx = getattr(roi, "takeoff_idx", None)
    landing_idx = getattr(roi, "landing_idx", None)
    flight = None
    if takeoff_idx is not None and landing_idx is not None:
        flight = safe_span((int(takeoff_idx), int(landing_idx)), n_samples)

    return {
        "braking": braking,
        "p1": p1,
        "p2": p2,
        "flight": flight,
        "landing": landing,
    }


def compute_roi_window_from_spans(
    t: np.ndarray,
    spans: List[Optional[Tuple[int, int]]],
    left_pad_ratio: float = 0.10,
    right_pad_ratio: float = 0.05,
    fallback_pad_left: int = 8,
    fallback_pad_right: int = 5,
) -> Tuple[float, float]:
    """
    Compute ROI x-window from a list of spans.
    """
    idx = []
    for sp in spans:
        if sp is None:
            continue
        s, e = sp
        idx.extend([s, e])

    if not idx:
        return float(t[0]), float(t[-1])

    s = max(0, min(idx))
    e = min(len(t) - 1, max(idx))

    if e > s:
        pad_left = int(left_pad_ratio * (e - s))
        pad_right = int(right_pad_ratio * (e - s))
    else:
        pad_left = fallback_pad_left
        pad_right = fallback_pad_right

    s = max(0, s - pad_left)
    e = min(len(t) - 1, e + pad_right)

    return float(t[s]), float(t[e])


def draw_global_phase_bands(
    ax,
    t: np.ndarray,
    phase_bands: Dict[str, Any],
    spans: Dict[str, Optional[Tuple[int, int]]],
    theme,
) -> None:
    """
    Draw global CMJ phase bands across the full visible axis height.
    """
    clear_artist_dict(phase_bands)

    bands = [
        ("Exzentrische Phase", theme.band_eccentric, spans.get("eccentric")),
        ("Konzentrische Phase", theme.band_concentric, spans.get("concentric")),
        ("Flugphase", theme.band_flight, spans.get("flight")),
        ("Landephase", theme.band_landing, spans.get("landing")),
    ]

    for name, color, span in bands:
        if span is None:
            continue

        s, e = span
        x0 = float(t[s])
        x1 = float(t[e - 1]) if e - 1 < len(t) else float(t[s])

        band = ax.axvspan(
            x0,
            x1,
            facecolor=color,
            alpha=0.10,
            zorder=0,
            linewidth=0.0,
        )
        phase_bands[name] = band


def draw_detailed_roi_shading(
    ax,
    t: np.ndarray,
    plot_force: np.ndarray,
    shaded: Dict[str, Any],
    spans: Dict[str, Optional[Tuple[int, int]]],
    theme,
) -> None:
    """
    Draw detailed CMJ phase shading as area under the total force curve.
    """
    clear_artist_dict(shaded)

    phases = [
        ("Bremsphase", theme.roi_braking, spans.get("braking")),
        ("Abstoßphase 1", theme.roi_p1, spans.get("p1")),
        ("Abstoßphase 2", theme.roi_p2, spans.get("p2")),
        ("Flugphase_Detail", theme.roi_flight, spans.get("flight")),
        ("Landephase_Detail", theme.roi_landing, spans.get("landing")),
    ]

    baseline = 0.0

    for name, color, span in phases:
        if span is None:
            continue

        s, e = span
        x = np.asarray(t[s:e], dtype=float)
        y = np.asarray(plot_force[s:e], dtype=float)

        poly = ax.fill_between(
            x,
            y,
            baseline,
            where=np.isfinite(y),
            interpolate=True,
            color=color,
            alpha=0.18,
            zorder=1,
        )

        try:
            poly.set_clip_on(True)
        except Exception:
            pass

        shaded[name] = poly


def add_event_line(
    ax,
    x_s: float,
    text: str,
    y_pos_ax: float,
    theme,
    artists_store: List[Any],
) -> None:
    """
    Draw a subtle vertical marker line + a small label.
    """
    ln = ax.axvline(
        x_s,
        color=theme.spine,
        alpha=0.55,
        linewidth=1.0,
        linestyle="-",
        label="_nolegend_",
        zorder=3,
    )
    ln.set_clip_on(True)
    artists_store.append(ln)

    txt = ax.text(
        x_s,
        y_pos_ax,
        text,
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color=theme.fg,
        alpha=0.85,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="#ffffff",
            edgecolor=theme.spine,
            alpha=0.85,
        ),
        zorder=4,
        clip_on=True,
    )
    artists_store.append(txt)


def annotate_phase_arrow(
    ax,
    t: np.ndarray,
    span: Optional[Tuple[int, int]],
    text: str,
    y_ax: float,
    color: str,
    edge_color: str,
    fg_color: str,
    artists_store: List[Any],
    is_allowed_cb,
    fontsize: int = 9,
) -> None:
    """
    Draw a horizontal double-arrow phase annotation below the curves.
    """
    if ax is None or span is None:
        return

    s, e = span
    if e <= s:
        return

    x0 = float(t[s])
    x1 = float(t[e - 1])
    x_mid = 0.5 * (x0 + x1)

    if not is_allowed_cb(x_mid):
        return

    ann = ax.annotate(
        "",
        xy=(x1, y_ax),
        xytext=(x0, y_ax),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(
            arrowstyle="<->",
            color=color,
            lw=1.2,
            shrinkA=0,
            shrinkB=0,
            alpha=0.95,
        ),
        zorder=6,
        annotation_clip=True,
    )
    artists_store.append(ann)

    txt = ax.text(
        x_mid,
        y_ax,
        text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=fg_color,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="#ffffff",
            edgecolor=edge_color,
            alpha=0.92,
        ),
        zorder=7,
        clip_on=True,
    )
    artists_store.append(txt)


def add_phase_arrows(
    ax,
    t: np.ndarray,
    spans: Dict[str, Optional[Tuple[int, int]]],
    theme,
    artists_store: List[Any],
    is_allowed_cb,
) -> None:
    """
    Draw the main phase arrows below the curves.
    """
    annotate_phase_arrow(
        ax=ax,
        t=t,
        span=spans.get("eccentric"),
        text="Exzentrische Phase",
        y_ax=0.04,
        color=theme.band_eccentric,
        edge_color=theme.spine,
        fg_color=theme.fg,
        artists_store=artists_store,
        is_allowed_cb=is_allowed_cb,
    )
    annotate_phase_arrow(
        ax=ax,
        t=t,
        span=spans.get("concentric"),
        text="Konzentrische Phase",
        y_ax=0.04,
        color=theme.band_concentric,
        edge_color=theme.spine,
        fg_color=theme.fg,
        artists_store=artists_store,
        is_allowed_cb=is_allowed_cb,
    )
    annotate_phase_arrow(
        ax=ax,
        t=t,
        span=spans.get("flight"),
        text="Flugphase",
        y_ax=0.04,
        color=theme.band_flight,
        edge_color=theme.spine,
        fg_color=theme.fg,
        artists_store=artists_store,
        is_allowed_cb=is_allowed_cb,
    )
    annotate_phase_arrow(
        ax=ax,
        t=t,
        span=spans.get("landing"),
        text="Landephase",
        y_ax=0.04,
        color=theme.band_landing,
        edge_color=theme.spine,
        fg_color=theme.fg,
        artists_store=artists_store,
        is_allowed_cb=is_allowed_cb,
    )


def style_axes(ax, fig, theme) -> None:
    """Apply axes styling."""
    ax.set_facecolor(theme.bg)
    if fig is not None:
        fig.patch.set_facecolor("#ffffff")
    ax.tick_params(colors=theme.fg)
    ax.xaxis.label.set_color(theme.fg)
    ax.yaxis.label.set_color(theme.fg)
    ax.title.set_color(theme.fg)
    for spine in ax.spines.values():
        spine.set_color(theme.spine)


def apply_grid(ax, theme) -> None:
    """Apply grid styling."""
    ax.grid(True, which="both", alpha=0.25, color=theme.grid)


def autoscale_y_visible(ax, lines: Dict[str, Any]) -> None:
    """Autoscale Y using visible Line2D only; ignore shading and bands."""
    ys = []
    for obj in lines.values():
        if not getattr(obj, "get_visible", lambda: True)():
            continue
        if isinstance(obj, (mcoll.PolyCollection, mpatches.Patch)):
            continue
        if isinstance(obj, mlines.Line2D):
            y = obj.get_ydata()
            if y is not None and len(y) > 0:
                ys.append(np.asarray(y, dtype=float))

    if not ys:
        return

    y_all = np.concatenate(ys)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        return

    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    pad = (y_max - y_min) * 0.08 if not np.isclose(y_min, y_max) else (1.0 if y_min == 0 else abs(y_min) * 0.05)
    ax.set_ylim(y_min - pad, y_max + pad)