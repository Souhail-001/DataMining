"""Decoupage du signal en fenetres temporelles (Partie IV, Question 7).

Fenetre de 2 s avec 50 % de recouvrement a 256 Hz :
    taille = 2 x 256 = 512 echantillons
    pas    = 512 x (1 - 0,5) = 256 echantillons, soit une nouvelle fenetre par seconde

Une fenetre est de classe 1 si au moins 50 % de ses echantillons sont en crise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["Windows", "window_parameters", "make_windows"]


@dataclass
class Windows:
    """Fenetres d'un enregistrement.

    data : tableau (n_fenetres, n_echantillons_par_fenetre, n_canaux). C'est une
           vue sur le signal : aucune copie memoire n'est faite.
    info : une ligne par fenetre (debut, fin, part d'echantillons en crise, Class).
    """

    data: np.ndarray
    info: pd.DataFrame
    sampling_rate: float
    channels: list[str]

    def __len__(self) -> int:
        return len(self.info)


def window_parameters(
    sampling_rate: float, window_s: float = 2.0, overlap: float = 0.5
) -> tuple[int, int]:
    """Taille et pas d'une fenetre, en echantillons."""
    if not 0 <= overlap < 1:
        raise ValueError("Le recouvrement doit etre dans [0, 1[.")

    size = int(round(window_s * sampling_rate))
    step = int(round(size * (1 - overlap)))
    return size, max(step, 1)


def make_windows(
    signal: np.ndarray,
    labels: np.ndarray,
    sampling_rate: float,
    channels: list[str],
    window_s: float = 2.0,
    overlap: float = 0.5,
    threshold: float = 0.5,
) -> Windows:
    """Decoupe un signal (n_echantillons, n_canaux) en fenetres etiquetees."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, None]

    size, step = window_parameters(sampling_rate, window_s, overlap)

    # (n_fenetres, n_canaux, size) -> (n_fenetres, size, n_canaux)
    views = sliding_window_view(signal, size, axis=0)[::step]
    views = np.swapaxes(views, 1, 2)

    label_views = sliding_window_view(np.asarray(labels, dtype=float), size)[::step]
    fraction = label_views.mean(axis=1)

    starts = np.arange(len(views)) * step / sampling_rate

    info = pd.DataFrame(
        {
            "window": np.arange(1, len(views) + 1),
            "start_s": starts,
            "end_s": starts + size / sampling_rate,
            "crisis_fraction": fraction.round(3),
            "Class": (fraction >= threshold).astype(int),
        }
    )

    return Windows(data=views, info=info, sampling_rate=sampling_rate, channels=list(channels))
