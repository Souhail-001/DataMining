"""Construction des modeles (Partie VIII, Question 12).

Protocole
---------
1. Separation stratifiee 80 % entrainement / 20 % test (`split_data`).
2. Pour chaque modele, les hyperparametres sont choisis par validation croisee
   stratifiee a 5 plis **sur l'entrainement seulement**, en maximisant le
   F1-score de la classe crise (`tune_models`). Le jeu de test n'est utilise
   qu'une fois, a la fin, pour la Partie IX.

Pourquoi le F1 et pas l'accuracy pour choisir ? Avec ~4 % de fenetres en
crise, un modele qui repond toujours "pas de crise" a deja 96 % d'accuracy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURES

__all__ = [
    "RANDOM_STATE",
    "POSITIVE_FEATURES",
    "LogTransformer",
    "split_data",
    "model_catalog",
    "TunedModel",
    "tune_models",
    "tuning_summary",
    "depth_curve",
    "UnderSampledClassifier",
    "undersampled_catalog",
]

RANDOM_STATE = 42

# Caracteristiques toujours positives et tres asymetriques (puissances,
# energies) : elles s'etalent sur plusieurs ordres de grandeur.
POSITIVE_FEATURES = ["Std", "Variance", "Amplitude", "RMS", "Energy", "Delta", "Theta", "Alpha", "Beta"]


class LogTransformer(BaseEstimator, TransformerMixin):
    """Passe les caracteristiques positives en log10.

    KNN et SVM reposent sur des distances : sans cette etape, quelques fenetres
    d'energie enorme ecrasent toutes les autres, meme apres standardisation.
    Les arbres n'en ont pas besoin (ils ne regardent que l'ordre des valeurs).
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for column in self.columns or POSITIVE_FEATURES:
            if column in X.columns:
                X[column] = np.log10(np.clip(X[column].to_numpy(dtype=float), 1e-12, None))
        return X


def split_data(
    dataset: pd.DataFrame,
    features: list[str] = FEATURES,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Separation stratifiee : la proportion de crises est la meme dans les deux jeux."""
    X = dataset[features]
    y = dataset["Class"].astype(int)

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def _scaled(model) -> Pipeline:
    return Pipeline([("log", LogTransformer()), ("scale", StandardScaler()), ("model", model)])


def model_catalog(random_state: int = RANDOM_STATE) -> dict:
    """Modeles compares et grilles d'hyperparametres explorees.

    Retourne {nom: (estimateur, grille)}.
    """
    tree_grid = {
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "class_weight": [None, "balanced"],
    }

    return {
        "Decision Tree (Gini)": (
            DecisionTreeClassifier(criterion="gini", random_state=random_state),
            tree_grid,
        ),
        "Decision Tree (Entropie)": (
            DecisionTreeClassifier(criterion="entropy", random_state=random_state),
            tree_grid,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=random_state),
            {
                "min_samples_leaf": [1, 3],
                "class_weight": [None, "balanced_subsample"],
            },
        ),
        "KNN": (
            _scaled(KNeighborsClassifier()),
            {
                "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "SVM (RBF)": (
            _scaled(SVC(kernel="rbf")),
            {
                "model__C": [1, 10, 100],
                "model__gamma": ["scale", 0.03, 0.3],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(max_iter=300, random_state=random_state),
            {
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "min_samples_leaf": [20, 50],
                "class_weight": [None, "balanced"],
            },
        ),
    }


@dataclass
class TunedModel:
    name: str
    estimator: object  # meilleur modele, re-entraine sur tout l'entrainement
    best_params: dict
    cv_f1: float
    cv_f1_std: float
    fit_time_s: float
    n_candidates: int
    search: object = field(repr=False, default=None)


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    catalog: dict | None = None,
    n_splits: int = 5,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> dict[str, TunedModel]:
    """Recherche sur grille + validation croisee pour chaque modele.

    Une reference naive (toujours "pas de crise") est ajoutee pour montrer ce
    que vaut l'accuracy seule.
    """
    catalog = catalog or model_catalog(random_state)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}
    for name, (estimator, grid) in catalog.items():
        search = GridSearchCV(estimator, grid, scoring=scoring, cv=folds, n_jobs=-1, refit=True)

        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        best = search.best_index_
        results[name] = TunedModel(
            name=name,
            estimator=search.best_estimator_,
            best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
            cv_f1=float(search.cv_results_["mean_test_score"][best]),
            cv_f1_std=float(search.cv_results_["std_test_score"][best]),
            fit_time_s=float(search.refit_time_),
            n_candidates=len(search.cv_results_["params"]),
            search=search,
        )
        if verbose:
            print(f"{name:26s} F1 (CV) = {results[name].cv_f1:.3f}   recherche : {elapsed:5.1f} s")

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    results["Reference : toujours 0"] = TunedModel(
        name="Reference : toujours 0",
        estimator=baseline,
        best_params={"strategy": "most_frequent"},
        cv_f1=0.0,
        cv_f1_std=0.0,
        fit_time_s=0.0,
        n_candidates=1,
    )

    return results


def tuning_summary(results: dict[str, TunedModel]) -> pd.DataFrame:
    """Hyperparametres retenus et score de validation croisee."""
    rows = []
    for name, tuned in results.items():
        params = ", ".join(f"{k}={v}" for k, v in tuned.best_params.items())
        rows.append(
            {
                "modele": name,
                "combinaisons testees": tuned.n_candidates,
                "hyperparametres retenus": params,
                "F1 (CV 5 plis)": f"{tuned.cv_f1:.3f} ± {tuned.cv_f1_std:.3f}",
                "temps d'entrainement (s)": round(tuned.fit_time_s, 2),
            }
        )
    return pd.DataFrame(rows)


def depth_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    depths=(1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None),
    criteria=("gini", "entropy"),
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """F1 en entrainement et en validation croisee selon la profondeur de l'arbre.

    Sert a comparer Gini et entropie a complexite egale, et a voir a partir de
    quelle profondeur l'arbre apprend par coeur (surapprentissage).
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for criterion in criteria:
        for depth in depths:
            tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=random_state)
            scores = cross_validate(tree, X_train, y_train, cv=folds, scoring="f1", return_train_score=True)
            rows.append(
                {
                    "criterion": criterion,
                    "max_depth": "aucune" if depth is None else depth,
                    "F1 entrainement": scores["train_score"].mean(),
                    "F1 validation": scores["test_score"].mean(),
                    "ecart-type validation": scores["test_score"].std(),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sous-echantillonnage (Partie XII : traiter le desequilibre des classes)
# ---------------------------------------------------------------------------
class UnderSampledClassifier(ClassifierMixin, BaseEstimator):
    """Entraine `estimator` sur un entrainement reequilibre.

    A chaque `fit`, on garde toutes les fenetres de crise et seulement
    `ratio` x (nombre de crises) fenetres hors crise, tirees au hasard.
    Les donnees de validation et de test ne sont jamais touchees : dans une
    validation croisee, seul le pli d'entrainement est sous-echantillonne.
    """

    def __init__(self, estimator=None, ratio=1.0, random_state=RANDOM_STATE):
        self.estimator = estimator
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_neg = min(len(neg), int(round(self.ratio * len(pos))))
        keep = np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
        X_kept = X.iloc[keep] if hasattr(X, "iloc") else np.asarray(X)[keep]
        self.estimator_ = clone(self.estimator).fit(X_kept, y[keep])
        self.classes_ = self.estimator_.classes_
        self.kept_ = {"crise": int(len(pos)), "hors_crise": int(n_neg)}
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


def undersampled_catalog(ratio: float = 1.0, random_state: int = RANDOM_STATE) -> dict:
    """Memes modeles et memes grilles que `model_catalog`, entraines apres sous-echantillonnage.

    `class_weight` est retire des grilles : le reequilibrage est deja fait
    par le tirage, on ne le compense pas une deuxieme fois.
    """
    catalog = {}
    for name, (estimator, grid) in model_catalog(random_state).items():
        grid = {f"estimator__{k}": v for k, v in grid.items() if not k.endswith("class_weight")}
        catalog[name] = (UnderSampledClassifier(estimator, ratio=ratio, random_state=random_state), grid)
    return catalog
