"""Analyse des erreurs du meilleur modele (Partie XI).

Chaque fenetre recoit un type : vrai positif (VP), faux positif (FP),
faux negatif (FN) ou vrai negatif (VN). On cherche ensuite **ou** se
trouvent les erreurs : au bord des crises, dans les crises courtes, chez
certains patients, ou sur des artefacts loin de toute crise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import match_recording_id
from .labeling import seizure_intervals

__all__ = [
    "ERROR_TYPES",
    "error_types",
    "error_counts",
    "add_error_context",
    "false_negative_profile",
    "false_positive_profile",
    "errors_by_recording",
    "plot_error_analysis",
]

ERROR_TYPES = ["VP", "FN", "FP", "VN"]
ERROR_NAMES = {
    "VP": "Vrai positif (crise detectee)",
    "FN": "Faux negatif (crise manquee)",
    "FP": "Faux positif (fausse alarme)",
    "VN": "Vrai negatif",
}
COLORS = {"VP": "#1baf7a", "FN": "#eb6834", "FP": "#eda100", "VN": "#2a78d6"}
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def error_types(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = np.empty(y_true.size, dtype=object)
    out[(y_true == 1) & (y_pred == 1)] = "VP"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    out[(y_true == 0) & (y_pred == 0)] = "VN"
    return out


def error_counts(types) -> pd.DataFrame:
    counts = pd.Series(types).value_counts().reindex(ERROR_TYPES, fill_value=0)
    return pd.DataFrame({
        "type": [ERROR_NAMES[t] for t in ERROR_TYPES],
        "fenetres": counts.to_numpy(),
        "part (%)": (100 * counts / counts.sum()).round(2).to_numpy(),
    }, index=ERROR_TYPES)


def add_error_context(windows: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Ajoute a chaque fenetre la crise la plus proche et sa distance (s).

    - `distance_crise_s` : 0 si la fenetre touche une crise, sinon l'ecart
      entre la fenetre et la crise la plus proche ;
    - `duree_crise_s` : duree de la crise touchee (NaN sinon).
    """
    frames = []
    for recording_id, part in windows.groupby("recording_id", sort=False):
        part = part.copy()
        true = seizure_intervals(annotations, recording_id)
        s = part["start_s"].to_numpy()[:, None]
        e = part["end_s"].to_numpy()[:, None]
        if len(true):
            gap = np.maximum(0, np.maximum(true[:, 0][None, :] - e, s - true[:, 1][None, :]))
            nearest = gap.argmin(axis=1)
            part["distance_crise_s"] = gap.min(axis=1)
            durations = (true[:, 1] - true[:, 0])[nearest]
            part["duree_crise_s"] = np.where(part["distance_crise_s"] == 0, durations, np.nan)
        else:
            part["distance_crise_s"] = np.inf
            part["duree_crise_s"] = np.nan
        frames.append(part)
    return pd.concat(frames).loc[windows.index]


def false_negative_profile(windows: pd.DataFrame) -> pd.DataFrame:
    """Taux de crises manquees selon la part de la fenetre en crise et la duree de la crise."""
    crisis = windows[windows["Class"] == 1].copy()
    crisis["part de la fenetre en crise"] = pd.cut(
        crisis["crisis_fraction"], [0.499, 0.75, 0.999, 1.0],
        labels=["50-75 % (bord)", "75-99 % (bord)", "100 % (coeur)"])
    crisis["duree de la crise"] = pd.cut(
        crisis["duree_crise_s"], [0, 5, 10, 20, 120], labels=["< 5 s", "5-10 s", "10-20 s", "> 20 s"])

    rows = []
    for column in ["part de la fenetre en crise", "duree de la crise"]:
        grouped = crisis.groupby(column, observed=True)["type"]
        for value, types in grouped:
            rows.append({
                "critere": column,
                "groupe": str(value),
                "fenetres de crise": len(types),
                "manquees (FN)": int((types == "FN").sum()),
                "taux de FN (%)": round(100 * (types == "FN").mean(), 1),
            })
    return pd.DataFrame(rows).set_index(["critere", "groupe"])


def false_positive_profile(windows: pd.DataFrame) -> pd.DataFrame:
    """Ou tombent les fausses alarmes : contre une crise, pres d'une crise, ou loin."""
    normal = windows[windows["Class"] == 0].copy()
    normal["position"] = pd.cut(
        normal["distance_crise_s"], [-0.1, 0.0, 10, np.inf],
        labels=["touche une crise (< 50 %)", "a moins de 10 s d'une crise", "loin de toute crise"])
    rows = []
    for value, types in normal.groupby("position", observed=True)["type"]:
        rows.append({
            "position de la fenetre": str(value),
            "fenetres hors crise": len(types),
            "fausses alarmes (FP)": int((types == "FP").sum()),
            "taux de FP (%)": round(100 * (types == "FP").mean(), 2),
        })
    table = pd.DataFrame(rows).set_index("position de la fenetre")
    table["part des FP (%)"] = (100 * table["fausses alarmes (FP)"] / table["fausses alarmes (FP)"].sum()).round(1)
    return table


def errors_by_recording(windows: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(windows["recording_id"].map(match_recording_id), windows["type"])
    table = table.reindex(columns=ERROR_TYPES, fill_value=0)
    table["Recall (%)"] = (100 * table["VP"] / (table["VP"] + table["FN"])).round(1)
    table["Precision (%)"] = (100 * table["VP"] / (table["VP"] + table["FP"])).round(1)
    return table.sort_values("FP", ascending=False)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_error_analysis(fn_profile: pd.DataFrame, fp_profile: pd.DataFrame, by_recording: pd.DataFrame):
    """Trois vues : FN selon la position dans la crise, FP selon la position, erreurs par patient."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})

    part = fn_profile.loc["part de la fenetre en crise"]
    axes[0].barh(part.index, part["taux de FN (%)"], color=COLORS["FN"])
    for i, v in enumerate(part["taux de FN (%)"]):
        axes[0].text(v + 1, i, f"{v:.0f} %", va="center", fontsize=9)
    axes[0].set_xlabel("Fenetres de crise manquees (%)")
    axes[0].set_title("Faux negatifs : surtout au bord des crises", fontsize=10, loc="left")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(100, part["taux de FN (%)"].max() + 15))

    axes[1].barh(fp_profile.index, fp_profile["part des FP (%)"], color=COLORS["FP"])
    for i, v in enumerate(fp_profile["part des FP (%)"]):
        axes[1].text(v + 1, i, f"{v:.0f} %", va="center", fontsize=9)
    axes[1].set_xlabel("Part des fausses alarmes (%)")
    axes[1].set_title("Faux positifs : ou tombent-ils ?", fontsize=10, loc="left")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)

    data = by_recording[["FN", "FP"]].sort_values("FP")
    y = np.arange(len(data))
    axes[2].barh(y - 0.2, data["FN"], height=0.4, color=COLORS["FN"], label="FN (crises manquees)")
    axes[2].barh(y + 0.2, data["FP"], height=0.4, color=COLORS["FP"], label="FP (fausses alarmes)")
    axes[2].set_yticks(y, data.index, fontsize=8)
    axes[2].set_xlabel("Nombre de fenetres")
    axes[2].set_title("Erreurs par enregistrement", fontsize=10, loc="left")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
    return fig
