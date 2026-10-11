"""Merge touching positive test windows without bridging untested gaps."""
import numpy as np
import pandas as pd


def detected_intervals(test_windows, predictions):
    ordered = test_windows.assign(Prediction=np.asarray(predictions)).sort_values('start_s')
    intervals = []
    for row in ordered.itertuples():
        if row.Prediction != 1:
            continue
        if intervals and row.start_s <= intervals[-1][1] + 1e-9:
            intervals[-1][1] = max(intervals[-1][1], row.end_s)
        else:
            intervals.append([row.start_s, row.end_s])
    result = pd.DataFrame(intervals, columns=['start_s', 'end_s'])
    result['duration_s'] = result.end_s - result.start_s
    return result


def compare_intervals(detected, actual):
    """Compare each prediction to the real interval with greatest temporal IoU."""
    rows = []
    for row in detected.itertuples():
        intersection = np.maximum(0, np.minimum(row.end_s, actual.end_s) - np.maximum(row.start_s, actual.start_s))
        union = row.duration_s + (actual.end_s - actual.start_s) - intersection
        iou = intersection / union
        best = iou.idxmax()
        rows.append({'predicted_start_s': row.start_s, 'predicted_end_s': row.end_s,
                     'actual_start_s': actual.loc[best, 'start_s'] if iou.loc[best] > 0 else np.nan,
                     'actual_end_s': actual.loc[best, 'end_s'] if iou.loc[best] > 0 else np.nan,
                     'IoU': iou.loc[best]})
    return pd.DataFrame(rows, columns=['predicted_start_s', 'predicted_end_s', 'actual_start_s', 'actual_end_s', 'IoU'])
