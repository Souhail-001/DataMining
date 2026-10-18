"""Detection temporelle des crises (Partie X, Question 14).

Le modele predit une classe par fenetre de 2 s. Pour retrouver des crises
dans le temps, on regroupe les fenetres consecutives predites "crise" en
intervalles, puis on compare ces intervalles aux annotations.

Pourquoi des predictions "hors enregistrement" ?
-----------------------------------------------
Le test 80/20 de la Partie VIII tire les fenetres au hasard : il ne reste
qu'une fenetre sur cinq dans chaque enregistrement, on ne peut donc pas
reconstruire une crise continue. Ici chaque enregistrement est predit par
un modele entraine sur les **autres** enregistrements (GroupKFold, 5 plis).
Toutes les fenetres sont alors des fenetres de test, comme pour un nouveau
patient.

Convention de temps
-------------------
Les fenetres font 2 s avec un pas de 1 s : la fenetre qui commence a `s`
couvre [s, s + 2]. On lui attribue sa seconde centrale [s + 0,5 ; s + 1,5].
Une suite de fenetres predites crise, de la fenetre a a la fenetre b, donne
donc l'intervalle [s_a + 0,5 ; s_b + 1,5].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .data import Recording, match_recording_id
from .labeling import seizure_intervals

__all__ = [
    "out_of_fold_proba",
    "smooth_proba",
    "postprocess",
    "predictions_to_intervals",
    "detected_intervals",
    "match_events",
    "event_summary",
    "interval_table",
    "plot_detection",
    "plot_detection_zoom",
]

TRUE_COLOR = "#eb6834"   # crise annotee
PRED_COLOR = "#2a78d6"   # crise predite
SIGNAL_COLOR = "#1f3a5f"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def out_of_fold_proba(estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                      n_splits: int = 5) -> np.ndarray:
    """Probabilite de crise de chaque fenetre, predite sans avoir vu son enregistrement."""
    proba = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return proba


def smooth_proba(proba, groups, size: int = 3) -> np.ndarray:
    """Moyenne glissante centree de la probabilite, enregistrement par enregistrement.

    Une crise dure au moins 2 s : une fenetre isolee predite crise au milieu de
    fenetres calmes est plus souvent un artefact qu'une vraie crise.
    """
    series = pd.Series(np.asarray(proba, dtype=float))
    smoothed = series.groupby(np.asarray(groups)).transform(
        lambda s: s.rolling(size, center=True, min_periods=1).mean()
    )
    return smoothed.to_numpy()


def postprocess(pred, groups, max_gap: int = 1, min_windows: int = 2) -> np.ndarray:
    """Regles de duree appliquees aux predictions, enregistrement par enregistrement.

    1. un trou de `max_gap` fenetre(s) au milieu d'une detection est comble
       (une crise ne s'arrete pas 1 s pour reprendre) ;
    2. une detection de moins de `min_windows` fenetres consecutives est
       supprimee (une fenetre isolee = 1 s, plus court que la plus courte crise, 2 s).
    """
    pred = np.asarray(pred).astype(int)
    out = pred.copy()
    for _, index in pd.Series(np.arange(pred.size)).groupby(np.asarray(groups)):
        idx = index.to_numpy()
        p = pred[idx].copy()
        ones = np.flatnonzero(p)
        for a, b in zip(ones[:-1], ones[1:]):
            if 1 < b - a <= max_gap + 1:
                p[a:b] = 1
        i = 0
        while i < p.size:
            if p[i]:
                j = i
                while j + 1 < p.size and p[j + 1]:
                    j += 1
                if j - i + 1 < min_windows:
                    p[i:j + 1] = 0
                i = j + 1
            else:
                i += 1
        out[idx] = p
    return out


# ---------------------------------------------------------------------------
# Fenetres -> intervalles
# ---------------------------------------------------------------------------
def predictions_to_intervals(pred, start_s, step_s: float = 1.0, window_s: float = 2.0) -> np.ndarray:
    """Regroupe les fenetres consecutives predites 1 en intervalles [debut, fin] (s)."""
    pred = np.asarray(pred).astype(bool)
    start_s = np.asarray(start_s, dtype=float)
    margin = (window_s - step_s) / 2  # 0,5 s pour 2 s / pas 1 s

    intervals = []
    i = 0
    while i < pred.size:
        if pred[i]:
            j = i
            while j + 1 < pred.size and pred[j + 1]:
                j += 1
            intervals.append((start_s[i] + margin, start_s[j] + window_s - margin))
            i = j + 1
        else:
            i += 1
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def detected_intervals(windows: pd.DataFrame, pred_column: str) -> dict[str, np.ndarray]:
    """Intervalles detectes pour chaque enregistrement ({recording_id: tableau (n, 2)})."""
    out = {}
    for recording_id, part in windows.groupby("recording_id", sort=True):
        part = part.sort_values("start_s")
        out[recording_id] = predictions_to_intervals(part[pred_column], part["start_s"])
    return out


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def match_events(true: np.ndarray, detected: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare crises annotees et crises detectees d'un enregistrement.

    Une crise annotee est **detectee** si au moins un intervalle predit la
    chevauche. Un intervalle predit qui ne chevauche aucune crise est une
    **fausse alarme**.

    Retourne (une ligne par crise annotee, une ligne par intervalle detecte).
    """
    true = np.asarray(true, dtype=float).reshape(-1, 2)
    detected = np.asarray(detected, dtype=float).reshape(-1, 2)

    rows_true = []
    for k, (start, end) in enumerate(true, start=1):
        hits = [(a, b) for a, b in detected if _overlap(start, end, a, b) > 0]
        covered = sum(_overlap(start, end, a, b) for a, b in hits)
        rows_true.append({
            "crise": k,
            "debut reel (s)": start,
            "fin reelle (s)": end,
            "duree (s)": round(end - start, 2),
            "detectee": bool(hits),
            "debut detecte (s)": hits[0][0] if hits else np.nan,
            "fin detectee (s)": hits[-1][1] if hits else np.nan,
            "retard au debut (s)": round(hits[0][0] - start, 2) if hits else np.nan,
            "ecart a la fin (s)": round(hits[-1][1] - end, 2) if hits else np.nan,
            "part couverte (%)": round(100 * covered / (end - start), 1) if end > start else np.nan,
            "morceaux": len(hits),
        })

    rows_det = []
    for k, (a, b) in enumerate(detected, start=1):
        matched = [i + 1 for i, (s, e) in enumerate(true) if _overlap(s, e, a, b) > 0]
        rows_det.append({
            "detection": k,
            "debut (s)": a,
            "fin (s)": b,
            "duree (s)": round(b - a, 2),
            "crise reelle": ", ".join(map(str, matched)) if matched else "aucune",
            "fausse alarme": not matched,
        })

    return pd.DataFrame(rows_true), pd.DataFrame(rows_det)


def event_summary(windows: pd.DataFrame, pred_column: str, annotations: pd.DataFrame) -> pd.DataFrame:
    """Bilan crise par crise, pour chaque enregistrement."""
    rows = []
    for recording_id, detected in detected_intervals(windows, pred_column).items():
        true = seizure_intervals(annotations, recording_id)
        per_true, per_det = match_events(true, detected)
        hours = windows.loc[windows["recording_id"] == recording_id, "end_s"].max() / 3600
        delays = per_true.loc[per_true["detectee"], "retard au debut (s)"] if len(per_true) else pd.Series(dtype=float)
        rows.append({
            "enregistrement": match_recording_id(recording_id),
            "crises reelles": len(true),
            "crises detectees": int(per_true["detectee"].sum()) if len(per_true) else 0,
            "intervalles predits": len(detected),
            "fausses alarmes": int(per_det["fausse alarme"].sum()) if len(per_det) else 0,
            "duree (h)": round(hours, 2),
            "retard median (s)": round(float(delays.median()), 1) if len(delays) else np.nan,
        })
    table = pd.DataFrame(rows).set_index("enregistrement")
    total = table.sum(numeric_only=True)
    total["retard median (s)"] = np.nan
    table.loc["TOTAL"] = total
    table["sensibilite (%)"] = (100 * table["crises detectees"] / table["crises reelles"]).round(1)
    table["fausses alarmes / h"] = (table["fausses alarmes"] / table["duree (h)"]).round(2)
    return table


def interval_table(detected: np.ndarray) -> pd.DataFrame:
    """Format demande par l'enonce : "Crise detectee k : debut -> fin"."""
    return pd.DataFrame({
        "Crise detectee": np.arange(1, len(detected) + 1),
        "Debut (s)": np.round(detected[:, 0], 1),
        "Fin (s)": np.round(detected[:, 1], 1),
        "Duree (s)": np.round(detected[:, 1] - detected[:, 0], 1),
    }).set_index("Crise detectee")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _envelope(t, x, max_points=30000):
    """Min/max par bloc : garde les pics quand on affiche une heure de signal."""
    if t.size <= max_points:
        return t, x
    block = int(np.ceil(t.size / (max_points / 2)))
    n = (t.size // block) * block
    tb = t[:n].reshape(-1, block)
    xb = x[:n].reshape(-1, block)
    tt = np.repeat(tb[:, 0], 2)
    xx = np.column_stack([xb.min(axis=1), xb.max(axis=1)]).ravel()
    return tt, xx


def _timeline(ax, true, detected, start, end):
    for a, b in true:
        ax.broken_barh([(a, b - a)], (1.15, 0.7), color=TRUE_COLOR)
    for a, b in detected:
        ax.broken_barh([(a, b - a)], (0.15, 0.7), color=PRED_COLOR)
    ax.set_yticks([0.5, 1.5], ["Predite", "Reelle"])
    ax.set_ylim(0, 2)
    ax.set_xlim(start, end)
    ax.set_xlabel("Temps (s)", fontsize=10)
    _style(ax)
    ax.grid(axis="y", visible=False)


def plot_detection(recording: Recording, channel: str, true, detected, proba=None, proba_time=None,
                   threshold: float = 0.5, start=None, end=None, title: str | None = None):
    """Signal EEG + crises reelles + crises predites (+ probabilite du modele)."""
    time = recording.time
    start = float(time[0]) if start is None else start
    end = float(time[-1]) if end is None else end
    mask = (time >= start) & (time <= end)
    t, x = _envelope(time[mask], recording.channel(channel)[mask])

    n_rows = 3 if proba is not None else 2
    ratios = [4, 1.6, 1.1] if proba is not None else [4, 1.1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 5.6 if proba is not None else 4.6), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    ax = axes[0]
    for a, b in np.asarray(true).reshape(-1, 2):
        ax.axvspan(a, b, color=TRUE_COLOR, alpha=0.18, linewidth=0)
    ax.plot(t, x, color=SIGNAL_COLOR, linewidth=0.5)
    spread = float(np.percentile(np.abs(x - np.median(x)), 99.5)) or 1.0
    ax.set_ylim(-1.3 * spread, 1.3 * spread)
    ax.set_ylabel(f"{channel} (µV)", fontsize=10)
    ax.set_title(title or f"{recording.recording_id} : signal, crises reelles et predites", fontsize=11, loc="left")
    _style(ax)

    if proba is not None:
        axp = axes[1]
        axp.plot(proba_time, proba, color=PRED_COLOR, linewidth=1)
        axp.axhline(threshold, color=INK_SECONDARY, linestyle="--", linewidth=0.8)
        axp.text(end, threshold, f" seuil {threshold:g}", va="center", ha="left", fontsize=8, color=INK_SECONDARY)
        axp.set_ylim(0, 1.02)
        axp.set_ylabel("P(crise)", fontsize=9)
        _style(axp)

    _timeline(axes[-1], np.asarray(true).reshape(-1, 2), np.asarray(detected).reshape(-1, 2), start, end)
    fig.tight_layout()
    return fig


def plot_detection_zoom(recording: Recording, channels: list[str], true, detected, start, end,
                        title: str | None = None):
    """Zoom sur quelques canaux autour d'une crise."""
    time = recording.time
    mask = (time >= start) & (time <= end)
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.8), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax = axes[0]
    offset = 0.0
    spacing = None
    ticks = []
    for name in channels:
        x = recording.channel(name)[mask]
        if spacing is None:
            spacing = 2.5 * float(np.percentile(np.abs(x - np.median(x)), 99)) or 1.0
        ax.plot(time[mask], x - offset, color=SIGNAL_COLOR, linewidth=0.6)
        ticks.append((-offset, name.replace("EEG", "")))
        offset += spacing
    for a, b in np.asarray(true).reshape(-1, 2):
        ax.axvspan(a, b, color=TRUE_COLOR, alpha=0.18, linewidth=0)
    ax.set_yticks([p for p, _ in ticks], [n for _, n in ticks])
    ax.set_title(title or f"{recording.recording_id} : zoom {start:.0f}-{end:.0f} s", fontsize=11, loc="left")
    _style(ax)
    _timeline(axes[1], np.asarray(true).reshape(-1, 2), np.asarray(detected).reshape(-1, 2), start, end)
    fig.tight_layout()
    return fig
