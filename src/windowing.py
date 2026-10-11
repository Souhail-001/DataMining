"""Cut a signal into complete windows and apply the 50% labeling rule."""
import numpy as np
import pandas as pd


def make_windows(values, labels, sampling_rate, seconds=2.0, overlap=0.5, start_time=0.0):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    if values.ndim != 1 or values.shape != labels.shape:
        raise ValueError('Use one channel and one label per sample.')
    if not np.isfinite(values).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Signal must be finite and labels must be binary.')
    if sampling_rate <= 0 or seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid sampling rate, window duration or overlap.')
    size = round(seconds * sampling_rate)
    step = round(size * (1 - overlap))
    if size < 2 or step < 1 or len(values) < size:
        raise ValueError('Window size is incompatible with the signal.')
    starts = np.arange(0, len(values) - size + 1, step)
    windows = np.array([values[i:i + size] for i in starts])
    fractions = np.array([labels[i:i + size].mean() for i in starts])
    metadata = pd.DataFrame({
        'start_s': start_time + starts / sampling_rate,
        'end_s': start_time + (starts + size) / sampling_rate,
        'seizure_fraction': fractions,
        'Class': (fractions >= 0.5).astype(int),
    })
    return windows, metadata
