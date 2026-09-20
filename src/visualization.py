"""Figures du mini-projet EEG (Question 2 et suivantes).

Toutes les fonctions retournent l'axe matplotlib utilise, ce qui permet de les
combiner dans une meme figure depuis le notebook.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import Recording

__all__ = [
    "amplitude_label",
    "plot_signal",
    "plot_overview",
    "plot_montage",
    "plot_events",
    "plot_spectrum",
]

# Palette volontairement sobre : une couleur de trace, une couleur d'accent.
SIGNAL_COLOR = "#1f3a5f"
ACCENT_COLOR = "#c2410c"
GRID_COLOR = "#d6dce2"

# Unite conventionnelle en EEG : le fichier MATLAB ne stocke pas l'unite.
DEFAULT_AMPLITUDE_UNIT = "µV"


def amplitude_label(recording: Recording) -> str:
    """Libelle de l'axe des amplitudes."""
    unit = recording.amplitude_unit or DEFAULT_AMPLITUDE_UNIT
    return f"Amplitude ({unit})"


def _style(ax: plt.Axes) -> plt.Axes:
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9aa5b1")

    ax.tick_params(labelsize=9, color="#9aa5b1")
    return ax


def _envelope(time: np.ndarray, values: np.ndarray, max_points: int):
    """Reduit le nombre de points traces en conservant les min/max locaux.

    Un sous-echantillonnage simple effacerait les pointes, qui sont justement
    le motif recherche dans un EEG de crise.
    """
    n = values.size

    if max_points <= 0 or n <= max_points:
        return time, values

    bucket = int(np.ceil(n / (max_points / 2)))
    usable = (n // bucket) * bucket

    blocks_t = time[:usable].reshape(-1, bucket)
    blocks_y = values[:usable].reshape(-1, bucket)

    centers = blocks_t.mean(axis=1)
    lows, highs = blocks_y.min(axis=1), blocks_y.max(axis=1)

    if usable < n:
        centers = np.append(centers, time[usable:].mean())
        lows = np.append(lows, values[usable:].min())
        highs = np.append(highs, values[usable:].max())

    out_time = np.repeat(centers, 2)
    out_values = np.empty(out_time.size)
    out_values[0::2], out_values[1::2] = lows, highs

    return out_time, out_values


def plot_events(
    ax: plt.Axes,
    events: pd.DataFrame,
    start: float | None = None,
    end: float | None = None,
    annotate: bool = True,
    max_labels: int = 12,
) -> plt.Axes:
    """Trace les marqueurs du manipulateur (YO, YF, HPN, SLI, ...)."""
    if events is None or events.empty:
        return ax

    visible = events
    if start is not None:
        visible = visible[visible["onset_s"] >= start]
    if end is not None:
        visible = visible[visible["onset_s"] <= end]

    if visible.empty:
        return ax

    for onset in visible["onset_s"]:
        ax.axvline(onset, color=ACCENT_COLOR, linewidth=0.7, alpha=0.5, zorder=1)

    if annotate and len(visible) <= max_labels:
        top = ax.get_ylim()[1]
        for onset, label in zip(visible["onset_s"], visible["label"]):
            ax.text(
                onset,
                top,
                f" {label}",
                rotation=90,
                fontsize=7,
                color=ACCENT_COLOR,
                va="top",
                ha="left",
            )

    return ax


def plot_signal(
    recording: Recording,
    channel: str | None = None,
    start: float | None = None,
    end: float | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_events: bool = False,
    max_points: int = 20000,
    ylim: str | tuple[float, float] = "auto",
    color: str = SIGNAL_COLOR,
    linewidth: float = 0.7,
) -> plt.Axes:
    """Trace un canal en fonction du temps (Question 2).

    L'axe horizontal est le temps en secondes, l'axe vertical l'amplitude.
    `ylim="robust"` cadre l'axe vertical sur 99,5 % des valeurs : les artefacts
    de quelques milliers de microvolts n'ecrasent plus le reste du trace.
    """
    channel = channel or recording.eeg_channels[0]
    window = recording.window(start, end)

    if window.empty:
        raise ValueError(f"Fenetre vide : verifiez start={start} et end={end}.")

    time = window[recording.time_column].to_numpy()
    values = window[channel].to_numpy()
    drawn_time, drawn_values = _envelope(time, values, max_points)

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.2))

    ax.plot(drawn_time, drawn_values, color=color, linewidth=linewidth, zorder=2)

    span = f"{time[0]:.0f}-{time[-1]:.0f} s"
    ax.set_title(
        title or f"Enregistrement {recording.recording_id} — canal {channel} ({span})",
        fontsize=11,
        loc="left",
    )
    ax.set_xlabel("Temps (s)", fontsize=10)
    ax.set_ylabel(amplitude_label(recording), fontsize=10)
    ax.set_xlim(time[0], time[-1])

    if ylim == "robust":
        center = float(np.median(values))
        spread = float(np.percentile(np.abs(values - center), 99.5)) or 1.0
        ax.set_ylim(center - 1.2 * spread, center + 1.2 * spread)
    elif isinstance(ylim, (tuple, list)):
        ax.set_ylim(*ylim)

    if show_events:
        plot_events(ax, recording.events, time[0], time[-1])

    return _style(ax)


def plot_overview(
    recording: Recording,
    channel: str | None = None,
    show_events: bool = True,
    ylim: str | tuple[float, float] = "robust",
    **kwargs,
) -> plt.Axes:
    """Vue d'ensemble de l'enregistrement complet (axe vertical robuste)."""
    channel = channel or recording.eeg_channels[0]

    return plot_signal(
        recording,
        channel=channel,
        show_events=show_events,
        ylim=ylim,
        title=(
            f"Enregistrement {recording.recording_id} — canal {channel} "
            f"— vue complete ({recording.duration:.0f} s)"
        ),
        **kwargs,
    )


def plot_montage(
    recording: Recording,
    channels: Sequence[str] | None = None,
    start: float = 0.0,
    end: float = 10.0,
    spacing: float | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_events: bool = False,
) -> plt.Axes:
    """Affiche plusieurs canaux empiles, comme sur un trace EEG clinique.

    Chaque canal est centre sur sa moyenne dans la fenetre, puis decale
    verticalement d'un pas constant.
    """
    channels = list(channels or recording.eeg_channels)
    window = recording.window(start, end)

    if window.empty:
        raise ValueError(f"Fenetre vide : verifiez start={start} et end={end}.")

    time = window[recording.time_column].to_numpy()
    centered = {name: window[name].to_numpy() - window[name].mean() for name in channels}

    if spacing is None:
        amplitude = np.median([np.percentile(np.abs(v), 95) for v in centered.values()])
        spacing = float(3.0 * amplitude) or 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 0.42 * len(channels) + 1.6))

    offsets = []
    for index, name in enumerate(channels):
        offset = -index * spacing
        offsets.append(offset)
        ax.plot(time, centered[name] + offset, color=SIGNAL_COLOR, linewidth=0.6)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channels, fontsize=8)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(offsets[-1] - spacing, spacing)

    ax.set_title(
        title
        or (
            f"Enregistrement {recording.recording_id} — {len(channels)} canaux "
            f"— {time[0]:.0f} a {time[-1]:.0f} s"
        ),
        fontsize=11,
        loc="left",
    )
    ax.set_xlabel("Temps (s)", fontsize=10)
    ax.set_ylabel(f"Canaux (decalage vertical : {spacing:.0f} {DEFAULT_AMPLITUDE_UNIT})", fontsize=9)

    if show_events:
        plot_events(ax, recording.events, time[0], time[-1], annotate=False)

    _style(ax)
    ax.grid(axis="y", visible=False)

    return ax


def plot_spectrum(
    recording: Recording,
    channel: str | None = None,
    start: float | None = None,
    end: float | None = None,
    f_max: float = 60.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
    highlight: float | None = 50.0,
) -> plt.Axes:
    """Densite spectrale de puissance (methode de Welch).

    Sert a justifier le pretraitement : la raie du secteur y est visible d'un
    coup d'oeil.
    """
    from scipy.signal import welch

    channel = channel or recording.eeg_channels[0]
    window = recording.window(start, end)
    values = window[channel].to_numpy()

    freqs, power = welch(
        values,
        fs=recording.sampling_rate,
        nperseg=min(int(4 * recording.sampling_rate), values.size),
    )
    visible = freqs <= f_max

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.2))

    ax.semilogy(freqs[visible], power[visible], color=SIGNAL_COLOR, linewidth=1.1)

    if highlight is not None and highlight <= f_max:
        ax.axvline(highlight, color=ACCENT_COLOR, linewidth=1.0, linestyle="--")
        ax.text(
            highlight,
            ax.get_ylim()[1],
            f" {highlight:.0f} Hz ",
            color=ACCENT_COLOR,
            fontsize=8,
            va="top",
            ha="right",
        )

    unit = recording.amplitude_unit or DEFAULT_AMPLITUDE_UNIT
    ax.set_title(title or f"Spectre \u2014 canal {channel}", fontsize=11, loc="left")
    ax.set_xlabel("Frequence (Hz)", fontsize=10)
    ax.set_ylabel(f"Densite spectrale ({unit}\u00b2/Hz)", fontsize=10)
    ax.set_xlim(0, f_max)

    return _style(ax)
