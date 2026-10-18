"""Amelioration du systeme (Partie XII).

Toutes les pistes sont evaluees de la meme facon : validation croisee **par
enregistrement** (GroupKFold, 5 plis), c'est-a-dire sur des patients que le
modele n'a jamais vus. C'est la mesure realiste (Partie IX), et la seule qui
permet de compter les crises detectees et les fausses alarmes dans le temps.

Le seuil de decision n'est jamais choisi sur le pli de test : dans chaque pli,
on le choisit par une seconde validation croisee (4 plis) faite uniquement sur
les enregistrements d'entrainement ("validation interne").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GroupKFold

from .detection import detected_intervals, match_events, smooth_proba
from .labeling import seizure_intervals

__all__ = [
    "THRESHOLDS",
    "add_context_features",
    "add_recording_normalization",
    "nested_threshold_predictions",
    "plain_predictions",
    "window_and_event_metrics",
    "threshold_curve",
    "plot_threshold_curve",
    "plot_improvements",
]

THRESHOLDS = np.round(np.arange(0.15, 0.71, 0.05), 2)
POSITIVE = ["Std", "Variance", "Amplitude", "RMS", "Energy", "Delta", "Theta", "Alpha", "Beta"]
BANDS = ["Delta", "Theta", "Alpha", "Beta"]


# ---------------------------------------------------------------------------
# Nouvelles caracteristiques
# ---------------------------------------------------------------------------
def add_context_features(dataset: pd.DataFrame, features: list[str], sizes=(3, 5)):
    """Ajoute, pour chaque caracteristique, sa moyenne sur les fenetres voisines.

    Une fenetre de 2 s ne voit qu'un morceau de crise ; ses voisines disent
    si l'activite dure. Les variables positives passent en log10 avant la
    moyenne pour qu'une seule fenetre enorme n'ecrase pas les autres.
    """
    frame = dataset.copy()
    base = frame[features].copy()
    for column in features:
        if column in POSITIVE:
            base[column] = np.log10(np.clip(base[column], 1e-12, None))
    new = []
    for size in sizes:
        rolled = base.groupby(frame["recording_id"]).transform(
            lambda s: s.rolling(size, center=True, min_periods=1).mean())
        for column in features:
            name = f"{column}_moy{size}"
            frame[name] = rolled[column]
            new.append(name)
    return frame, features + new


def add_recording_normalization(dataset: pd.DataFrame, features: list[str]):
    """Normalisation par enregistrement + puissances relatives.

    - chaque variable positive est divisee par sa mediane dans l'enregistrement
      (en log10 : on soustrait la mediane) : on mesure "combien de fois plus
      que d'habitude chez ce patient" ;
    - puissances relatives : Delta / (Delta + Theta + Alpha + Beta), etc.
    """
    frame = dataset.copy()
    total = frame[BANDS].sum(axis=1)
    new = []
    for band in BANDS:
        frame[f"{band}_rel"] = frame[band] / total
        new.append(f"{band}_rel")
    for column in features:
        if column in POSITIVE:
            log = np.log10(np.clip(frame[column], 1e-12, None))
            frame[f"{column}_norm"] = log - log.groupby(frame["recording_id"]).transform("median")
            new.append(f"{column}_norm")
    return frame, features + new


# ---------------------------------------------------------------------------
# Predictions par enregistrement
# ---------------------------------------------------------------------------
def _best_threshold(y, proba, thresholds=THRESHOLDS) -> float:
    scores = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def plain_predictions(estimator, X, y, groups, n_splits: int = 5) -> np.ndarray:
    """Probabilite hors enregistrement, sans reglage de seuil."""
    proba = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return proba


def nested_threshold_predictions(estimator, X, y, groups, smoothing_sizes=(1, 3),
                                 n_splits: int = 5, inner_splits: int = 4, verbose: bool = True) -> dict:
    """Probabilites hors enregistrement et seuil choisi en validation interne.

    Pour chaque pli externe :
      1. validation croisee interne (par enregistrement) sur l'entrainement
         -> probabilites "hors enregistrement" des fenetres d'entrainement ;
      2. pour chaque taille de lissage, le seuil qui maximise le F1 sur ces
         probabilites ;
      3. modele entraine sur tout l'entrainement, applique au pli de test avec
         ce seuil.

    Retourne {"proba": ..., size: {"pred": ..., "thresholds": [...]}}.
    """
    groups = pd.Series(np.asarray(groups), index=X.index)
    proba = np.zeros(len(y))
    result = {size: {"pred": np.zeros(len(y), dtype=int), "thresholds": []} for size in smoothing_sizes}

    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        X_tr, y_tr, g_tr = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]

        inner = np.zeros(len(train_idx))
        for in_tr, in_va in GroupKFold(n_splits=inner_splits).split(X_tr, y_tr, g_tr):
            model = clone(estimator).fit(X_tr.iloc[in_tr], y_tr.iloc[in_tr])
            inner[in_va] = model.predict_proba(X_tr.iloc[in_va])[:, 1]

        model = clone(estimator).fit(X_tr, y_tr)
        p_test = model.predict_proba(X.iloc[test_idx])[:, 1]
        proba[test_idx] = p_test

        for size in smoothing_sizes:
            inner_s = smooth_proba(inner, g_tr, size) if size > 1 else inner
            threshold = _best_threshold(y_tr.to_numpy(), inner_s)
            test_s = smooth_proba(p_test, groups.iloc[test_idx], size) if size > 1 else p_test
            result[size]["pred"][test_idx] = (test_s >= threshold).astype(int)
            result[size]["thresholds"].append(threshold)

        if verbose:
            chosen = ", ".join(f"lissage {s} : {result[s]['thresholds'][-1]:.2f}" for s in smoothing_sizes)
            print(f"pli {fold} : seuils choisis en validation interne -> {chosen}")

    result["proba"] = proba
    return result


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------
def window_and_event_metrics(windows: pd.DataFrame, y_pred, annotations: pd.DataFrame) -> dict:
    """Metriques par fenetre (Precision, Recall, F1) et par crise (sensibilite, fausses alarmes / h)."""
    y_true = windows["Class"].to_numpy()
    frame = windows[["recording_id", "start_s", "end_s"]].copy()
    frame["pred"] = np.asarray(y_pred).astype(int)

    n_true = n_hit = n_false = 0
    hours = 0.0
    for recording_id, detected in detected_intervals(frame, "pred").items():
        per_true, per_det = match_events(seizure_intervals(annotations, recording_id), detected)
        n_true += len(per_true)
        n_hit += int(per_true["detectee"].sum()) if len(per_true) else 0
        n_false += int(per_det["fausse alarme"].sum()) if len(per_det) else 0
        hours += frame.loc[frame["recording_id"] == recording_id, "end_s"].max() / 3600

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "crises detectees": f"{n_hit}/{n_true}",
        "sensibilite crises (%)": round(100 * n_hit / n_true, 1),
        "fausses alarmes": n_false,
        "fausses alarmes / h": round(n_false / hours, 2),
    }


def threshold_curve(y, proba, thresholds=np.round(np.arange(0.05, 0.96, 0.05), 2)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({"seuil": t,
                     "Precision": precision_score(y, pred, zero_division=0),
                     "Recall": recall_score(y, pred, zero_division=0),
                     "F1-score": f1_score(y, pred, zero_division=0)})
    return pd.DataFrame(rows)


def plot_threshold_curve(curves: dict[str, pd.DataFrame]):
    """Precision, Recall et F1 selon le seuil, avec et sans lissage."""
    colors = {"Precision": "#2a78d6", "Recall": "#eb6834", "F1-score": "#1baf7a"}
    styles = ["--", "-"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for (label, curve), style in zip(curves.items(), styles):
        for metric, color in colors.items():
            ax.plot(curve["seuil"], curve[metric], color=color, linestyle=style, linewidth=1.8 if style == "-" else 1.2,
                    label=f"{metric} ({label})")
    ax.axvline(0.5, color="#52514e", linewidth=0.8, linestyle=":")
    ax.text(0.505, 0.03, "seuil par defaut 0,5", fontsize=8, color="#52514e")
    ax.set_xlabel("Seuil de decision sur P(crise)")
    ax.set_ylabel("Score (classe crise)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Random Forest, patients jamais vus : effet du seuil et du lissage", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_improvements(table: pd.DataFrame, baseline: str):
    """F1 par fenetre et fausses alarmes par heure pour chaque variante."""
    data = table.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(data) + 1.6), sharey=True)
    base_f1 = table.loc[baseline, "F1-score"]
    base_fa = table.loc[baseline, "fausses alarmes / h"]

    colors = ["#1baf7a" if v > base_f1 + 0.005 else ("#eb6834" if v < base_f1 - 0.005 else "#8a8986")
              for v in data["F1-score"]]
    axes[0].barh(data.index, data["F1-score"], color=colors)
    axes[0].axvline(base_f1, color="#52514e", linestyle="--", linewidth=0.8)
    for i, v in enumerate(data["F1-score"]):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(0.5, 0.82)
    axes[0].set_xlabel("F1-score par fenetre (plus haut = mieux)")

    colors = ["#1baf7a" if v < base_fa - 0.1 else ("#eb6834" if v > base_fa + 0.1 else "#8a8986")
              for v in data["fausses alarmes / h"]]
    axes[1].barh(data.index, data["fausses alarmes / h"], color=colors)
    axes[1].axvline(base_fa, color="#52514e", linestyle="--", linewidth=0.8)
    for i, v in enumerate(data["fausses alarmes / h"]):
        axes[1].text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=9)
    axes[1].set_xlabel("Fausses alarmes par heure (plus bas = mieux)")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="x", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Partie XII : effet de chaque amelioration (validation par enregistrement)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig
