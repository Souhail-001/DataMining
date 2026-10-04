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
    "plot_band_powers",
    "CLASS_COLORS",
    "shade_intervals",
    "plot_class_signal",
    "plot_windowing",
    "plot_window_fft",
]

# Palette volontairement sobre : une couleur de trace, une couleur d'accent.
SIGNAL_COLOR = "#1f3a5f"
ACCENT_COLOR = "#c2410c"
GRID_COLOR = "#d6dce2"

# Unite conventionnelle en EEG : le fichier MATLAB ne stocke pas l'unite.
DEFAULT_AMPLITUDE_UNIT = "µV"

# Rampe sequentielle (clair -> fonce) pour les bandes rangees par frequence
# croissante : delta, theta, alpha, beta. La luminosite porte l'ordre, pas la
# teinte, donc la lecture reste correcte en niveaux de gris.
BAND_RAMP = ("#7b9dc4", "#5a80ab", "#3b5d8c", "#1f3a5f")

# Couleur des etiquettes : encre, jamais la couleur de la serie.
LABEL_COLOR = "#3f4a56"

# Certains marqueurs sont des releves d'impedance de plus de 180 caracteres :
# ecrits en entier a la verticale, ils deforment la figure. Le tableau des
# marqueurs reste la pour les lire en entier.
MAX_LABEL_CHARS = 24


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
    max_label_chars: int = MAX_LABEL_CHARS,
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
            if len(label) > max_label_chars:
                label = label[: max_label_chars - 1] + "…"

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


def plot_band_powers(
    powers: pd.Series,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Part de chaque bande de frequence dans la puissance totale.

    Les puissances absolues s'etalent sur plusieurs decades : sur un axe
    lineaire elles ecraseraient les bandes hautes. On trace donc la part
    relative, seule grandeur comparable d'une fenetre a l'autre, et chaque
    barre porte sa valeur.
    """
    names = list(powers.index)
    values = powers.to_numpy(dtype=float)
    total = float(values.sum())
    share = values / total * 100.0 if total > 0 else np.zeros_like(values)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.55 * len(names) + 1.5))

    # Premiere bande en haut : l'ordre des frequences se lit de haut en bas.
    positions = np.arange(len(names))[::-1]
    colors = [BAND_RAMP[index % len(BAND_RAMP)] for index in range(len(names))]

    ax.barh(positions, share, height=0.62, color=colors, zorder=2)

    ceiling = max(float(share.max()), 1.0)
    for position, percent in zip(positions, share):
        ax.text(
            percent + 0.02 * ceiling,
            position,
            f"{percent:.1f} %",
            va="center",
            fontsize=9,
            color=LABEL_COLOR,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, ceiling * 1.18)
    ax.set_xlabel("Part de la puissance totale (%)", fontsize=10)
    ax.set_title(title or "Puissance par bande", fontsize=11, loc="left")

    _style(ax)
    ax.grid(axis="y", visible=False)

    return ax


# ---------------------------------------------------------------------------
# Parties II a VI : crises annotees, classes, fenetres, spectre d'une fenetre
# ---------------------------------------------------------------------------
# Couleurs des classes (palette categorielle validee, ordre fixe).
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_NAMES = {0: "Classe 0 : hors crise", 1: "Classe 1 : crise"}


def shade_intervals(ax: plt.Axes, intervals, label: str | None = "Crise annotee",
                    color: str = CLASS_COLORS[1], alpha: float = 0.18) -> plt.Axes:
    """Colorie les intervalles [debut, fin] sur un axe temporel."""
    for index, (start, end) in enumerate(np.asarray(intervals).reshape(-1, 2)):
        ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0,
                   label=label if index == 0 else None, zorder=0)
    return ax


def plot_class_signal(
    recording: Recording,
    labels: np.ndarray,
    channel: str,
    start: float | None = None,
    end: float | None = None,
    title: str | None = None,
    ylim: str | tuple[float, float] = "robust",
    max_points: int = 40000,
) -> plt.Figure:
    """Question 6 : le signal colore selon la classe, et la colonne Class en dessous."""
    time = recording.time
    mask = np.ones(time.size, dtype=bool)
    if start is not None:
        mask &= time >= start
    if end is not None:
        mask &= time < end

    t, values, cls = time[mask], recording.channel(channel)[mask], np.asarray(labels)[mask]

    fig, (ax, ax_cls) = plt.subplots(
        2, 1, figsize=(13, 4.4), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    for value in (0, 1):
        # On trace chaque classe separement ; NaN coupe la ligne hors de la classe.
        y = np.where(cls == value, values, np.nan)
        tt, yy = _envelope(t, np.nan_to_num(y, nan=0.0), max_points)
        keep = _envelope(t, (cls == value).astype(float), max_points)[1] > 0
        ax.plot(tt, np.where(keep, yy, np.nan), color=CLASS_COLORS[value],
                linewidth=0.6 if value == 0 else 0.8, label=CLASS_NAMES[value])

    if ylim == "robust":
        spread = float(np.percentile(np.abs(values - np.median(values)), 99.5)) or 1.0
        ax.set_ylim(-1.3 * spread, 1.3 * spread)
    elif isinstance(ylim, (tuple, list)):
        ax.set_ylim(*ylim)

    ax.set_ylabel(amplitude_label(recording), fontsize=10)
    ax.set_title(title or f"{recording.recording_id} : canal {channel}", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    _style(ax)

    ax_cls.fill_between(t, cls, step="post", color=CLASS_COLORS[1], alpha=0.8, linewidth=0)
    ax_cls.set_ylim(-0.1, 1.2)
    ax_cls.set_yticks([0, 1])
    ax_cls.set_ylabel("Class", fontsize=9)
    ax_cls.set_xlabel("Temps (s)", fontsize=10)
    ax_cls.set_xlim(t[0], t[-1])
    _style(ax_cls)

    fig.tight_layout()
    return fig


def plot_windowing(
    recording: Recording,
    labels: np.ndarray,
    windows_info: pd.DataFrame,
    channel: str,
    start: float,
    end: float,
) -> plt.Figure:
    """Question 7 : fenetres de 2 s (pas de 1 s) autour du debut d'une crise."""
    fig, (ax, ax_w) = plt.subplots(
        2, 1, figsize=(13, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 3]}
    )

    window = recording.window(start, end)
    t = window[recording.time_column].to_numpy()
    lab = np.asarray(labels)[(recording.time >= start) & (recording.time < end)]
    ax.plot(t, window[channel], color=SIGNAL_COLOR, linewidth=0.7)
    ax.fill_between(t, 0, 1, where=lab > 0, transform=ax.get_xaxis_transform(),
                    color=CLASS_COLORS[1], alpha=0.18, linewidth=0, label="Echantillons en crise")
    ax.set_ylabel(amplitude_label(recording), fontsize=9)
    ax.set_title(f"Decoupage en fenetres de 2 s avec 50 % de recouvrement ({channel})",
                 fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax)

    shown = windows_info[(windows_info["start_s"] >= start) & (windows_info["end_s"] <= end)]
    for row, (_, w) in enumerate(shown.iterrows()):
        color = CLASS_COLORS[int(w["Class"])]
        ax_w.barh(row, w["end_s"] - w["start_s"], left=w["start_s"], height=0.7,
                  color=color, alpha=0.85)
        ax_w.text(w["end_s"] + 0.1, row, f"{w['crisis_fraction']:.0%}", va="center",
                  fontsize=7, color=LABEL_COLOR)

    ax_w.set_yticks([])
    ax_w.invert_yaxis()
    ax_w.set_ylabel("Fenetres\n(% en crise)", fontsize=9)
    ax_w.set_xlabel("Temps (s)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[k]) for k in (0, 1)]
    ax_w.legend(handles, ["Class = 0 (moins de 50 % en crise)", "Class = 1 (au moins 50 % en crise)"],
                loc="lower left", fontsize=8, frameon=False, ncol=2)
    ax_w.set_xlim(start, end)
    _style(ax_w)
    ax_w.grid(axis="y", visible=False)

    fig.tight_layout()
    return fig


def plot_window_fft(
    windows: np.ndarray,
    info: pd.DataFrame,
    indices: dict[str, int],
    channel_index: int,
    channel_name: str,
    sampling_rate: float,
    bands: dict[str, tuple[float, float]],
    f_max: float = 40.0,
) -> plt.Figure:
    """Partie VI : une fenetre hors crise et une fenetre de crise, en temps et en frequence."""
    from .features import fft_psd

    fig, axes = plt.subplots(len(indices), 2, figsize=(13, 3.0 * len(indices)),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    axes = np.atleast_2d(axes)

    for row, (label, index) in enumerate(indices.items()):
        x = windows[index, :, channel_index]
        cls = int(info.iloc[index]["Class"])
        color = CLASS_COLORS[cls]
        t = info.iloc[index]["start_s"] + np.arange(x.size) / sampling_rate

        ax_t, ax_f = axes[row]
        ax_t.plot(t, x, color=color, linewidth=0.8)
        ax_t.set_title(f"{label} : domaine temporel ({channel_name})", fontsize=10, loc="left")
        ax_t.set_xlabel("Temps (s)", fontsize=9)
        ax_t.set_ylabel(f"Amplitude ({DEFAULT_AMPLITUDE_UNIT})", fontsize=9)
        _style(ax_t)

        freqs, psd = fft_psd(x[None, :], sampling_rate, axis=1)
        psd = psd[0]
        visible = freqs <= f_max
        ax_f.plot(freqs[visible], psd[visible], color=color, linewidth=1.2)
        for k, (name, (low, high)) in enumerate(bands.items()):
            ax_f.axvspan(low, high, color=BAND_RAMP[k], alpha=0.12, linewidth=0)
            ax_f.text((low + high) / 2, 1.0, name, transform=ax_f.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=8, color=LABEL_COLOR)
        peak = freqs[visible][np.argmax(psd[visible])]
        ax_f.annotate(f"pic {peak:.1f} Hz", (peak, psd[visible].max()), xytext=(6, -12),
                      textcoords="offset points", fontsize=8, color=LABEL_COLOR)
        ax_f.set_xlim(0, f_max)
        ax_f.set_xlabel("Frequence (Hz)", fontsize=9)
        ax_f.set_ylabel(f"DSP ({DEFAULT_AMPLITUDE_UNIT}²/Hz)", fontsize=9)
        ax_f.set_title("Spectre (FFT)", fontsize=10, loc="left", pad=14)
        _style(ax_f)

    fig.tight_layout()
    return fig
