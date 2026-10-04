"""Statistiques des crises (Partie II) et analyse exploratoire des caracteristiques (Partie VII)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .visualization import CLASS_COLORS, CLASS_NAMES, LABEL_COLOR, _style

__all__ = [
    "seizure_table",
    "seizure_statistics",
    "plot_seizure_durations",
    "class_comparison",
    "plot_class_means",
    "plot_class_boxplots",
    "correlated_pairs",
    "plot_correlation",
]

INK_SECONDARY = "#52514e"


# ---------------------------------------------------------------------------
# Partie II
# ---------------------------------------------------------------------------
def seizure_table(annotations: pd.DataFrame, excel_id: str) -> pd.DataFrame:
    """Question 3 : Crise | Debut (s) | Fin (s) | Duree (s) pour un enregistrement."""
    rows = annotations[annotations["recording_id"] == excel_id].sort_values("start_s")
    return pd.DataFrame(
        {
            "Crise": np.arange(1, len(rows) + 1),
            "Debut (heure)": rows["start_clock"].to_numpy(),
            "Fin (heure)": rows["end_clock"].to_numpy(),
            "Debut (s)": rows["start_s"].to_numpy(),
            "Fin (s)": rows["end_s"].to_numpy(),
            "Duree (s)": rows["duration_s"].to_numpy(),
        }
    ).set_index("Crise")


def _intervals(group: pd.DataFrame) -> np.ndarray:
    """Temps sans crise entre la fin d'une crise et le debut de la suivante."""
    group = group.sort_values("start_s")
    return group["start_s"].to_numpy()[1:] - group["end_s"].to_numpy()[:-1]


def seizure_statistics(annotations: pd.DataFrame) -> pd.DataFrame:
    """Question 4 : durees et intervalles, par enregistrement puis sur l'ensemble."""
    rows = []
    for recording_id, group in annotations.groupby("recording_id"):
        gaps = _intervals(group)
        rows.append(
            {
                "enregistrement": recording_id,
                "crises": len(group),
                "duree moyenne (s)": group["duration_s"].mean(),
                "duree min (s)": group["duration_s"].min(),
                "duree max (s)": group["duration_s"].max(),
                "intervalle moyen (s)": gaps.mean() if gaps.size else np.nan,
            }
        )

    all_gaps = np.concatenate([_intervals(g) for _, g in annotations.groupby("recording_id")])
    rows.append(
        {
            "enregistrement": "TOUS",
            "crises": len(annotations),
            "duree moyenne (s)": annotations["duration_s"].mean(),
            "duree min (s)": annotations["duration_s"].min(),
            "duree max (s)": annotations["duration_s"].max(),
            "intervalle moyen (s)": all_gaps.mean(),
        }
    )
    return pd.DataFrame(rows).set_index("enregistrement").round(1)


def plot_seizure_durations(annotations: pd.DataFrame) -> plt.Figure:
    """Une ligne par enregistrement, un point par crise : dispersion des durees."""
    order = (
        annotations.groupby("recording_id")["duration_s"].median().sort_values().index.tolist()
    )
    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2, 1]})

    for row, recording_id in enumerate(order):
        values = annotations.loc[annotations["recording_id"] == recording_id, "duration_s"]
        ax.scatter(values, np.full(len(values), row), s=36, color=CLASS_COLORS[1],
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.plot([values.min(), values.max()], [row, row], color="#c9c8c3", linewidth=1, zorder=2)

    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("Duree de la crise (s)")
    ax.set_title("Duree de chaque crise, par enregistrement", fontsize=11, loc="left")
    _style(ax)

    durations = annotations["duration_s"]
    ax_h.hist(durations, bins=np.arange(0, durations.max() + 5, 5), color=CLASS_COLORS[1],
              edgecolor="white", linewidth=1.5)
    ax_h.axvline(durations.median(), color=LABEL_COLOR, linestyle="--", linewidth=1)
    ax_h.text(durations.median(), ax_h.get_ylim()[1] * 0.95, f"  mediane {durations.median():.0f} s",
              fontsize=8, color=LABEL_COLOR, va="top")
    ax_h.set_xlabel("Duree (s)")
    ax_h.set_ylabel("Nombre de crises")
    ax_h.set_title(f"Distribution ({len(durations)} crises)", fontsize=11, loc="left")
    _style(ax_h)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Partie VII
# ---------------------------------------------------------------------------
def class_comparison(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Question 10 et 11.3 : valeurs par classe et pouvoir discriminant.

    AUC univariee : probabilite qu'une fenetre de crise tiree au hasard ait une
    valeur plus grande qu'une fenetre hors crise. 0,5 = la caracteristique ne
    separe rien ; 1 (ou 0) = separation parfaite. On affiche max(AUC, 1 - AUC).
    """
    y = dataset["Class"].to_numpy()
    rows = []
    for feature in features:
        values = dataset[feature].to_numpy()
        auc = roc_auc_score(y, values)
        median_0 = np.median(values[y == 0])
        median_1 = np.median(values[y == 1])
        rows.append(
            {
                "caracteristique": feature,
                "moyenne classe 0": values[y == 0].mean(),
                "moyenne classe 1": values[y == 1].mean(),
                "mediane classe 0": median_0,
                "mediane classe 1": median_1,
                "rapport des medianes (1/0)": median_1 / median_0 if median_0 > 0 else np.nan,
                "AUC (pouvoir discriminant)": max(auc, 1 - auc),
            }
        )
    table = pd.DataFrame(rows).set_index("caracteristique")
    return table.sort_values("AUC (pouvoir discriminant)", ascending=False)


def plot_class_means(dataset: pd.DataFrame, features=("Energy", "Delta", "Theta", "Std")) -> plt.Figure:
    """Moyenne (barre) et mediane (trait) par classe pour les caracteristiques demandees."""
    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.6))

    for ax, feature in zip(np.atleast_1d(axes), features):
        means = dataset.groupby("Class")[feature].mean()
        medians = dataset.groupby("Class")[feature].median()
        ax.bar([0, 1], means, color=[CLASS_COLORS[0], CLASS_COLORS[1]], width=0.6)
        ax.scatter([0, 1], medians, marker="_", s=600, color=LABEL_COLOR, zorder=3,
                   label="mediane")
        ratio = means[1] / means[0]
        ax.set_title(f"{feature}\nmoyenne x{ratio:.0f} en crise", fontsize=10)
        ax.set_xticks([0, 1], ["hors crise", "crise"], fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        _style(ax)
        ax.grid(axis="x", visible=False)

    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Question 10 : moyenne par classe (barres) et mediane (trait)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_class_boxplots(dataset: pd.DataFrame, features: list[str], log: bool = True) -> plt.Figure:
    """Distribution de chaque caracteristique par classe (echelle log)."""
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        data = [dataset.loc[dataset["Class"] == k, feature].to_numpy() for k in (0, 1)]
        positive = all((d > 0).all() for d in data)
        parts = ax.boxplot(data, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                           medianprops={"color": LABEL_COLOR, "linewidth": 1.4})
        for patch, k in zip(parts["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[k])
            patch.set_alpha(0.75)
            patch.set_edgecolor(CLASS_COLORS[k])
        if log and positive:
            ax.set_yscale("log")
        ax.set_xticks([1, 2], ["hors crise", "crise"], fontsize=8)
        ax.set_title(feature + (" (log)" if log and positive else ""), fontsize=10)
        _style(ax)
        ax.grid(axis="x", visible=False)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle("Distribution des caracteristiques par classe (boites : 25-75 %, moustaches : 5-95 %)",
                 fontsize=11)
    fig.tight_layout()
    return fig


def correlated_pairs(corr: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Couples de caracteristiques dont |r| depasse le seuil."""
    pairs = []
    names = corr.columns
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({"caracteristique 1": names[i], "caracteristique 2": names[j], "r": r})
    return pd.DataFrame(pairs).sort_values("r", key=np.abs, ascending=False).reset_index(drop=True)


def plot_correlation(corr: pd.DataFrame, title: str) -> plt.Figure:
    """Carte de chaleur divergente (bleu = negatif, gris = 0, rouge = positif)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(8.2, 7))
    image = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)

    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.75 else "#0b0b0b")

    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="coefficient de correlation r")
    fig.tight_layout()
    return fig
