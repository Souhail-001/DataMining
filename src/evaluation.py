"""Evaluation des modeles (Partie IX).

Convention : la classe positive est 1 (crise).
  Precision = parmi les fenetres annoncees "crise", part de vraies crises
  Recall    = parmi les vraies crises, part de fenetres detectees
  F1        = moyenne harmonique des deux
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

__all__ = [
    "compute_metrics",
    "results_table",
    "confusion_table",
    "plot_confusion_matrices",
    "plot_metric_comparison",
    "compare_trees",
    "group_evaluation",
    "plot_depth_curve",
    "plot_feature_importances",
]

METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]

# Palette categorielle validee (ordre fixe) et encres de texte.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9893")
    ax.tick_params(labelsize=9, colors=INK_SECONDARY)
    return ax


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }


def results_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """Tableau de la Partie IX : une ligne par modele, trie par F1."""
    rows = []
    for name, tuned in models.items():
        estimator = getattr(tuned, "estimator", tuned)
        start = time.perf_counter()
        y_pred = estimator.predict(X_test)
        predict_ms = (time.perf_counter() - start) * 1000

        rows.append({"Modele": name, **compute_metrics(y_test, y_pred), "prediction (ms)": predict_ms})

    table = pd.DataFrame(rows).set_index("Modele")
    return table.sort_values("F1-score", ascending=False)


def confusion_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """VP, FP, FN, VN de chaque modele."""
    rows = []
    for name, tuned in models.items():
        estimator = getattr(tuned, "estimator", tuned)
        tn, fp, fn, tp = confusion_matrix(y_test, estimator.predict(X_test), labels=[0, 1]).ravel()
        rows.append(
            {
                "Modele": name,
                "VP (crise detectee)": tp,
                "FN (crise manquee)": fn,
                "FP (fausse alarme)": fp,
                "VN": tn,
            }
        )
    return pd.DataFrame(rows).set_index("Modele")


def plot_confusion_matrices(models: dict, X_test, y_test, ncols: int = 4):
    """Une matrice de confusion par modele, en effectifs et en % de la classe reelle."""
    names = list(models)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        estimator = getattr(models[name], "estimator", models[name])
        matrix = confusion_matrix(y_test, estimator.predict(X_test), labels=[0, 1])
        share = matrix / matrix.sum(axis=1, keepdims=True)

        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]}\n({share[i, j]:.0%})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if share[i, j] > 0.6 else INK,
                )
        ax.set_xticks([0, 1], ["0 hors crise", "1 crise"], fontsize=8)
        ax.set_yticks([0, 1], ["0 hors crise", "1 crise"], fontsize=8)
        ax.set_xlabel("Classe predite", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Classe reelle", fontsize=9, color=INK_SECONDARY)
        ax.set_title(name, fontsize=10, color=INK)

    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("Matrices de confusion sur le jeu de test (effectif et % de la ligne)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_metric_comparison(table: pd.DataFrame, metrics=METRICS):
    """Barres horizontales groupees : une couleur par metrique."""
    table = table.iloc[::-1]  # meilleur modele en haut
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(table) + 1.2))

    height = 0.8 / len(metrics)
    y = np.arange(len(table))
    for k, metric in enumerate(metrics):
        ax.barh(
            y + (k - (len(metrics) - 1) / 2) * height,
            table[metric],
            height=height * 0.9,
            color=SERIES[k],
            label=metric,
        )

    ax.set_yticks(y, table.index)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score sur le jeu de test (0 a 1)")
    ax.set_title("Comparaison des modeles (classe positive = crise)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def compare_trees(models: dict, names: list[str], X_test, y_test) -> pd.DataFrame:
    """Comparaison detaillee de deux arbres (Gini contre entropie)."""
    rows = {}
    predictions = {}
    for name in names:
        tuned = models[name]
        tree = tuned.estimator
        y_pred = tree.predict(X_test)
        predictions[name] = y_pred
        rows[name] = {
            **{k: round(v, 3) for k, v in compute_metrics(y_test, y_pred).items()},
            "F1 validation croisee": round(tuned.cv_f1, 3),
            "profondeur": tree.get_depth(),
            "nombre de feuilles": tree.get_n_leaves(),
            "temps d'entrainement (ms)": round(tuned.fit_time_s * 1000, 1),
            "hyperparametres": ", ".join(f"{k}={v}" for k, v in tuned.best_params.items()),
        }

    table = pd.DataFrame(rows)
    if len(names) == 2:
        agreement = (predictions[names[0]] == predictions[names[1]]).mean()
        table.loc["predictions identiques (%)"] = [round(agreement * 100, 2)] * 2
    return table


def group_evaluation(models: dict, X, y, groups, n_splits: int = 5) -> pd.DataFrame:
    """Validation croisee par enregistrement (GroupKFold).

    Toutes les fenetres d'un enregistrement vont soit en entrainement, soit en
    test. C'est la situation reelle : le modele doit reconnaitre les crises
    d'un patient qu'il n'a jamais vu. Les hyperparametres sont ceux retenus
    a la Partie VIII.
    """
    folds = GroupKFold(n_splits=n_splits)
    rows = []

    for name, tuned in models.items():
        estimator = getattr(tuned, "estimator", tuned)
        y_true_all, y_pred_all, fold_f1 = [], [], []

        for train_idx, test_idx in folds.split(X, y, groups):
            model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X.iloc[test_idx])
            y_true_all.append(y.iloc[test_idx].to_numpy())
            y_pred_all.append(y_pred)
            fold_f1.append(f1_score(y.iloc[test_idx], y_pred, zero_division=0))

        metrics = compute_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
        rows.append(
            {
                "Modele": name,
                **metrics,
                "F1 par pli (min-max)": f"{min(fold_f1):.2f} - {max(fold_f1):.2f}",
            }
        )

    return pd.DataFrame(rows).set_index("Modele").sort_values("F1-score", ascending=False)


def plot_depth_curve(curve: pd.DataFrame):
    """F1 selon la profondeur maximale, pour Gini et entropie."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = curve["max_depth"].astype(str).unique().tolist()
    x = np.arange(len(labels))
    names = {"gini": "Gini", "entropy": "Entropie"}

    for k, (criterion, group) in enumerate(curve.groupby("criterion", sort=False)):
        color = SERIES[k]
        ax.plot(x, group["F1 validation"], color=color, linewidth=2, marker="o", markersize=5,
                label=f"{names[criterion]} - validation croisee")
        ax.plot(x, group["F1 entrainement"], color=color, linewidth=1.5, linestyle="--",
                label=f"{names[criterion]} - entrainement")

    ax.set_xticks(x, labels)
    ax.set_xlabel("Profondeur maximale de l'arbre (max_depth)")
    ax.set_ylabel("F1-score (classe crise)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Arbre de decision : Gini contre entropie selon la profondeur")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_feature_importances(models: dict, names: list[str], features: list[str]):
    """Importance des caracteristiques (diminution d'impurete) pour plusieurs arbres/forets."""
    data = pd.DataFrame(
        {name: models[name].estimator.feature_importances_ for name in names}, index=features
    )
    data = data.loc[data.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(features) + 1.2))
    height = 0.8 / len(names)
    y = np.arange(len(data))
    for k, name in enumerate(names):
        ax.barh(y + (k - (len(names) - 1) / 2) * height, data[name], height=height * 0.9,
                color=SERIES[k], label=name)

    ax.set_yticks(y, data.index)
    ax.set_xlabel("Importance (part de la reduction d'impurete)")
    ax.set_title("Caracteristiques utilisees par les modeles a base d'arbres")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
