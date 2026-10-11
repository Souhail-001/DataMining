"""Questions 3–5: annotation summaries and sample-level target construction."""

import numpy as np
import pandas as pd

from .data import Recording, match_recording_id


def select_seizures(annotations: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    """Select a recording explicitly; missing annotations are not negative labels."""
    key = match_recording_id(recording_id)
    selected = annotations.loc[
        annotations["recording_id"].str.strip().str.upper() == key
    ].sort_values("start_s").reset_index(drop=True).copy()
    if selected.empty:
        raise ValueError(f"No seizure annotations found for {recording_id}.")
    counts = selected["reported_count"].dropna().unique()
    if len(counts) != 1 or counts[0] != len(selected):
        raise ValueError("Extracted seizure count does not match NombreCE.")
    validate_intervals(selected)
    selected["duration_s"] = selected["end_s"] - selected["start_s"]
    return selected


def validate_intervals(seizures: pd.DataFrame) -> None:
    """Reject ambiguous multi-recording tables, nonfinite or reversed intervals."""
    if "recording_id" in seizures and seizures["recording_id"].nunique() > 1:
        raise ValueError("Select one recording before processing intervals.")
    intervals = seizures[["start_s", "end_s"]].to_numpy(dtype=float)
    if not np.isfinite(intervals).all():
        raise ValueError("Seizure boundaries must be finite.")
    if np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("Seizure intervals must have nonnegative starts and positive duration.")


def seizure_statistics(seizures: pd.DataFrame) -> pd.DataFrame:
    """Q4: duration statistics and end-to-next-start gaps, all in seconds.

    Also report onset-to-onset intervals to remove ambiguity in 'between'.
    An undefined mean (fewer than two seizures) is NaN, never zero.
    """
    validate_intervals(seizures)
    ordered = seizures.sort_values("start_s")
    starts = ordered["start_s"].to_numpy(dtype=float)
    ends = ordered["end_s"].to_numpy(dtype=float)
    durations = ends - starts
    gaps = starts[1:] - ends[:-1]
    if np.any(gaps < 0):
        raise ValueError("Overlapping seizures: a seizure-free gap is undefined.")
    return pd.DataFrame([{
        "seizure_count": len(ordered),
        "mean_duration_s": durations.mean() if len(durations) else np.nan,
        "min_duration_s": durations.min() if len(durations) else np.nan,
        "max_duration_s": durations.max() if len(durations) else np.nan,
        "mean_gap_s": gaps.mean() if len(gaps) else np.nan,
        "mean_onset_interval_s": np.diff(starts).mean() if len(starts) > 1 else np.nan,
    }])


def label_samples(recording: Recording, seizures: pd.DataFrame) -> pd.DataFrame:
    """Q5: copy the raw table and add Class using closed [start, end] intervals.

    Inputs must already share the recording's elapsed-second reference.
    Intervals outside the acquisition are rejected, never clipped or shifted.
    Empty tables are rejected because absent annotations do not prove no seizures.
    """
    validate_intervals(seizures)
    if seizures.empty:
        raise ValueError("No verified annotations: cannot assign sample classes.")
    if "recording_id" in seizures:
        keys = seizures["recording_id"].str.strip().str.upper()
        if not keys.eq(match_recording_id(recording.recording_id)).all():
            raise ValueError("Annotations belong to a different recording.")
    time = recording.time
    if not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Sample times must be finite and strictly increasing.")
    acquisition_end = time[-1] + recording.sampling_period
    if (seizures.start_s < time[0]).any() or (seizures.end_s > acquisition_end).any():
        raise ValueError("Annotations exceed recording bounds; check units and time origin.")
    labels = np.zeros(len(time), dtype=np.uint8)
    for start, end in seizures[["start_s", "end_s"]].itertuples(index=False, name=None):
        left = np.searchsorted(time, start, side="left")
        right = np.searchsorted(time, end, side="right")
        if left == right:
            raise ValueError("A seizure contains no sampled instant; check annotation units.")
        labels[left:right] = 1
    result = recording.data.copy()
    result["Class"] = labels
    return result


def seizures_from_header(events: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Use explicit header seizure onsets and chronological Excel durations.

    This implements the requested source convention, without estimating a clock
    offset. Equal counts are required; ambiguous/missing markers are rejected.
    """
    validate_intervals(annotations)
    markers = events.loc[
        events.label.str.strip().str.fullmatch(r'absence|abscence|crise', case=False, na=False)
    ].sort_values('onset_s').reset_index(drop=True)
    if markers.empty or len(markers) != len(annotations):
        raise ValueError('Explicit header seizure markers must match the Excel seizure count.')
    result = annotations.sort_values('start_s').reset_index(drop=True).copy()
    result['excel_start_clock_s'] = result.start_s
    result['excel_end_clock_s'] = result.end_s
    result['duration_s'] = result.end_s - result.start_s
    result['start_s'] = markers.onset_s.to_numpy()
    result['end_s'] = result.start_s + result.duration_s
    result['header_label'] = markers.label.to_numpy()
    result['alignment_source'] = 'header onset + Excel duration; chronological pairing'
    validate_intervals(result)
    return result
