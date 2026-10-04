"""Caracteristiques des fenetres EEG (Partie V, Questions 7-8 et Partie VI, Question 9).

Chaque fenetre contient 19 canaux EEG. Pour obtenir UNE valeur par fenetre,
chaque caracteristique est calculee canal par canal, puis on prend la
**mediane sur les 19 canaux**. Une crise d'absence est generalisee (tous les
canaux changent en meme temps) alors qu'un artefact ou une saturation touche
peu de canaux : la mediane garde le premier et ignore le second.

Temporelles (Partie V)                 Frequentielles (Partie VI, par FFT)
  Mean      moyenne                      Delta  0,5-4 Hz
  Std       ecart-type                   Theta  4-8 Hz
  Variance  variance                     Alpha  8-13 Hz
  Min, Max  extremes                     Beta   13-30 Hz
  Amplitude Max - Min (crete a crete)
  RMS       racine de la moyenne des carres
  Energy    somme des carres
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import file_id, load_recording, raw_dir
from .labeling import label_samples, seizure_intervals
from .preprocessing import clean_recording
from .windowing import make_windows

__all__ = [
    "EEG_CHANNELS",
    "BANDS",
    "TEMPORAL_FEATURES",
    "SPECTRAL_FEATURES",
    "FEATURES",
    "temporal_features",
    "fft_psd",
    "band_powers_fft",
    "window_features",
    "recording_dataset",
    "build_dataset",
]

# Les 19 electrodes du systeme international 10-20, communes a tous les
# enregistrements annotes (l'ordre des colonnes change d'un fichier a l'autre,
# on les selectionne donc par leur nom).
EEG_CHANNELS = [
    "EEGFp1", "EEGFp2", "EEGF7", "EEGF3", "EEGFz", "EEGF4", "EEGF8",
    "EEGT3", "EEGC3", "EEGCz", "EEGC4", "EEGT4",
    "EEGT5", "EEGP3", "EEGPz", "EEGP4", "EEGT6", "EEGO1", "EEGO2",
]

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
}

TEMPORAL_FEATURES = ["Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy"]
SPECTRAL_FEATURES = list(BANDS)
FEATURES = TEMPORAL_FEATURES + SPECTRAL_FEATURES


def temporal_features(windows: np.ndarray) -> pd.DataFrame:
    """Caracteristiques temporelles, fenetres de forme (n, taille, canaux)."""
    x = np.asarray(windows, dtype=float)

    mean = x.mean(axis=1)
    var = x.var(axis=1)
    low, high = x.min(axis=1), x.max(axis=1)
    energy = np.einsum("nsc,nsc->nc", x, x)  # somme des carres, sans copie
    rms = np.sqrt(energy / x.shape[1])

    per_channel = {
        "Mean": mean,
        "Std": np.sqrt(var),
        "Variance": var,
        "Min": low,
        "Max": high,
        "Amplitude": high - low,
        "RMS": rms,
        "Energy": energy,
    }

    return pd.DataFrame({name: np.median(values, axis=1) for name, values in per_channel.items()})


def fft_psd(x: np.ndarray, sampling_rate: float, axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Densite spectrale de puissance par FFT (fenetre de Hann, spectre unilateral).

    Retourne (frequences en Hz, DSP en unite^2/Hz).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]

    shape = [1] * x.ndim
    shape[axis] = n
    taper = np.hanning(n).reshape(shape)

    # On retire la moyenne pour que la composante continue ne fuie pas dans la bande Delta.
    centered = x - x.mean(axis=axis, keepdims=True)
    spectrum = np.fft.rfft(centered * taper, axis=axis)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    psd = np.abs(spectrum) ** 2 / (sampling_rate * (taper**2).sum())
    psd = np.moveaxis(psd, axis, -1)
    psd[..., 1:-1] *= 2  # spectre unilateral : on replie les frequences negatives
    return freqs, np.moveaxis(psd, -1, axis)


def band_powers_fft(
    windows: np.ndarray, sampling_rate: float, bands: dict | None = None
) -> pd.DataFrame:
    """Puissance par bande (integrale de la DSP), mediane sur les canaux."""
    bands = bands or BANDS
    freqs, psd = fft_psd(windows, sampling_rate, axis=1)  # (n, f, canaux)
    df = freqs[1] - freqs[0]

    powers = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        per_channel = psd[:, mask, :].sum(axis=1) * df
        powers[name] = np.median(per_channel, axis=1)

    return pd.DataFrame(powers)


def window_features(windows: np.ndarray, sampling_rate: float, chunk: int = 500) -> pd.DataFrame:
    """Toutes les caracteristiques, calculees par paquets pour limiter la memoire."""
    parts = []
    for begin in range(0, len(windows), chunk):
        block = np.ascontiguousarray(windows[begin : begin + chunk])
        parts.append(
            pd.concat(
                [temporal_features(block), band_powers_fft(block, sampling_rate)], axis=1
            )
        )
    return pd.concat(parts, ignore_index=True)


def recording_dataset(
    recording_id: str,
    annotations: pd.DataFrame,
    root: Path | str | None = None,
    window_s: float = 2.0,
    overlap: float = 0.5,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Dataset d'un enregistrement : une ligne = une fenetre EEG."""
    recording = load_recording(recording_id, root, with_events=False)
    clean = clean_recording(recording, channels=EEG_CHANNELS)

    intervals = seizure_intervals(annotations, recording_id)
    labels = label_samples(clean.time, intervals)

    windows = make_windows(
        clean.data[EEG_CHANNELS].to_numpy(),
        labels,
        clean.sampling_rate,
        EEG_CHANNELS,
        window_s=window_s,
        overlap=overlap,
        threshold=threshold,
    )

    features = window_features(windows.data, clean.sampling_rate)
    frame = pd.concat([windows.info, features], axis=1)
    frame.insert(0, "recording_id", recording_id)

    return frame[["recording_id", "window", "start_s", "end_s", "crisis_fraction", *FEATURES, "Class"]]


def build_dataset(
    annotations: pd.DataFrame,
    root: Path | str | None = None,
    recording_ids: Iterable[str] | None = None,
    cache: Path | str | None = None,
    verbose: bool = True,
    rebuild: bool = False,
    **window_kwargs,
) -> pd.DataFrame:
    """Dataset complet sur tous les enregistrements annotes.

    Seuls les enregistrements presents a la fois dans le dossier `data/raw` et
    dans le fichier d'annotations sont utilises : sans annotation, on ne
    connait pas la classe des fenetres. Si `cache` est donne, le resultat est
    enregistre en CSV et relu aux appels suivants (sauf si `rebuild=True`).
    """
    if cache is not None and Path(cache).is_file() and not rebuild:
        return pd.read_csv(cache)

    folder = raw_dir(root)
    if recording_ids is None:
        recording_ids = [
            file_id(excel_id)
            for excel_id in sorted(annotations["recording_id"].unique())
            if (folder / f"{file_id(excel_id)}d.mat").is_file()
        ]

    frames = []
    for recording_id in recording_ids:
        frame = recording_dataset(recording_id, annotations, root, **window_kwargs)
        frames.append(frame)
        if verbose:
            print(
                f"{recording_id} : {len(frame):5d} fenetres, "
                f"{int(frame['Class'].sum()):4d} en crise"
            )

    dataset = pd.concat(frames, ignore_index=True)

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(cache, index=False)

    return dataset
