"""Eight temporal features and four FFT band powers for each window."""
import numpy as np
import pandas as pd

BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}


def fft_power(windows, sampling_rate):
    """One-sided periodogram; summing PSD * df gives mean-square AC power."""
    windows = np.atleast_2d(windows)
    n = windows.shape[1]
    centered = windows - windows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2 / (sampling_rate * n)
    power[:, 1:-1 if n % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(n, 1 / sampling_rate), power


def extract_features(windows, sampling_rate):
    x = np.asarray(windows, dtype=float)
    result = pd.DataFrame({
        'Mean': x.mean(axis=1), 'Std': x.std(axis=1),
        'Variance': x.var(axis=1), 'Min': x.min(axis=1),
        'Max': x.max(axis=1), 'Amplitude': np.ptp(x, axis=1),
        'RMS': np.sqrt(np.mean(x ** 2, axis=1)), 'Energy': np.sum(x ** 2, axis=1),
    })
    frequencies, power = fft_power(x, sampling_rate)
    df = sampling_rate / x.shape[1]
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        result[name] = power[:, mask].sum(axis=1) * df
    return result
