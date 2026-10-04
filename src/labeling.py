"""Construction de la variable cible (Partie III, Questions 5 et 6).

Chaque echantillon recoit une classe :
    0 -> hors crise
    1 -> pendant une crise, c'est-a-dire debut <= t <= fin pour une crise annotee
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Recording, match_recording_id

__all__ = [
    "seizure_intervals",
    "label_samples",
    "add_class_column",
    "class_balance",
]


def seizure_intervals(annotations: pd.DataFrame, recording_id: str) -> np.ndarray:
    """Intervalles [debut, fin] en secondes pour un enregistrement.

    `recording_id` peut etre donne au format fichier (230102B_F_0000) ou
    Excel (230102B-F).
    """
    key = recording_id if "-" in recording_id else match_recording_id(recording_id)
    rows = annotations[annotations["recording_id"] == key.upper()]
    rows = rows.sort_values("start_s")

    return rows[["start_s", "end_s"]].to_numpy(dtype=float).reshape(-1, 2)


def label_samples(time: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Classe de chaque instant : 1 si t appartient a un intervalle [debut, fin]."""
    time = np.asarray(time, dtype=float)
    labels = np.zeros(time.size, dtype=np.int8)

    for start, end in intervals:
        labels[(time >= start) & (time <= end)] = 1

    return labels


def add_class_column(recording: Recording, intervals: np.ndarray) -> pd.DataFrame:
    """Copie du signal avec la colonne `Class` ajoutee en derniere position."""
    frame = recording.data.copy()
    frame["Class"] = label_samples(recording.time, intervals)
    return frame


def class_balance(labels: np.ndarray | pd.Series) -> pd.DataFrame:
    """Effectif et pourcentage de chaque classe."""
    labels = pd.Series(np.asarray(labels)).astype(int)
    counts = labels.value_counts().reindex([0, 1], fill_value=0)

    return pd.DataFrame(
        {
            "classe": ["0 - hors crise", "1 - crise"],
            "effectif": counts.to_numpy(),
            "pourcentage": (counts.to_numpy() / counts.sum() * 100).round(2),
        }
    )
