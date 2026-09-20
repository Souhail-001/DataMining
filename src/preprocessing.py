"""Pretraitement du signal EEG (etape "Pretraitement" du pipeline).

Le signal brut de ces enregistrements est domine par le 50 Hz du secteur et par
une derive lente : sans filtrage, ni l'oeil ni les caracteristiques calculees
plus tard ne voient l'activite cerebrale. Ce module fournit le minimum
necessaire : un coupe-bande sur le secteur et un passe-bande 0,5-40 Hz.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, welch

from .data import Recording

__all__ = [
    "mains_frequency",
    "notch_filter",
    "bandpass_filter",
    "clean_signal",
    "clean_recording",
    "band_powers",
]

DEFAULT_BAND = (0.5, 40.0)


def mains_frequency(values: np.ndarray, sampling_rate: float) -> float:
    """Detecte la frequence du secteur presente dans le signal (50 ou 60 Hz)."""
    freqs, power = welch(values, fs=sampling_rate, nperseg=min(4096, values.size))

    candidates = []
    for target in (50.0, 60.0):
        mask = np.abs(freqs - target) <= 1.0
        candidates.append(power[mask].max() if mask.any() else 0.0)

    return 50.0 if candidates[0] >= candidates[1] else 60.0


def notch_filter(
    values: np.ndarray,
    sampling_rate: float,
    frequency: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """Supprime une frequence isolee (interference du reseau electrique)."""
    b, a = iirnotch(w0=frequency, Q=quality, fs=sampling_rate)
    return filtfilt(b, a, values, axis=0)


def bandpass_filter(
    values: np.ndarray,
    sampling_rate: float,
    band: tuple[float, float] = DEFAULT_BAND,
    order: int = 4,
) -> np.ndarray:
    """Passe-bande de Butterworth applique sans dephasage (filtfilt)."""
    low, high = band
    nyquist = sampling_rate / 2.0

    if not 0 < low < high < nyquist:
        raise ValueError(f"Bande {band} incompatible avec fs = {sampling_rate} Hz.")

    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, values, axis=0)


def clean_signal(
    values: np.ndarray,
    sampling_rate: float,
    band: tuple[float, float] = DEFAULT_BAND,
    notch: float | None = 50.0,
) -> np.ndarray:
    """Coupe-bande secteur puis passe-bande."""
    cleaned = np.asarray(values, dtype=float)

    if notch is not None and notch < sampling_rate / 2:
        cleaned = notch_filter(cleaned, sampling_rate, notch)

    return bandpass_filter(cleaned, sampling_rate, band)


def clean_recording(
    recording: Recording,
    channels: Sequence[str] | None = None,
    band: tuple[float, float] = DEFAULT_BAND,
    notch: float | None = None,
) -> Recording:
    """Retourne une copie filtree de l'enregistrement.

    Par defaut la frequence du secteur est detectee sur le premier canal EEG.
    """
    channels = list(channels or recording.channels)

    if notch is None:
        notch = mains_frequency(recording.channel(recording.eeg_channels[0]), recording.sampling_rate)

    frame = recording.data.copy()
    frame[channels] = clean_signal(
        frame[channels].to_numpy(),
        recording.sampling_rate,
        band=band,
        notch=notch,
    )

    source = dict(recording.source)
    source["preprocessing"] = f"notch {notch:.0f} Hz + passe-bande {band[0]}-{band[1]} Hz"

    return replace(recording, data=frame, source=source)


def band_powers(
    values: np.ndarray,
    sampling_rate: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> pd.Series:
    """Puissance moyenne par bande de frequence (utile a partir de la Partie VI)."""
    bands = bands or {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }

    freqs, power = welch(values, fs=sampling_rate, nperseg=min(1024, values.size))

    return pd.Series(
        {
            name: float(power[(freqs >= low) & (freqs < high)].sum())
            for name, (low, high) in bands.items()
        }
    )
