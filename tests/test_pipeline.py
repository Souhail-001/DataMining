"""Checks for window boundaries, FFT units, splits and interval reconstruction."""
import unittest
import numpy as np
import pandas as pd
from src.data import align_clock_annotations
from src.windowing import make_windows
from src.labeling import seizures_from_header
from src.features import extract_features, fft_power
from src.training import train_models, purge_overlaps
from src.evaluation import evaluate_models
from src.detection import detected_intervals, compare_intervals


class PipelineTests(unittest.TestCase):
    def test_windows_threshold_overlap_and_tail(self):
        windows, meta = make_windows(np.arange(11), [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], 2, 2, .5)
        self.assertEqual(windows.shape, (4, 4))
        np.testing.assert_array_equal(meta.Class, [1, 1, 0, 0])
        np.testing.assert_array_equal(meta.start_s, [0, 1, 2, 3])
        self.assertEqual(meta.end_s.iloc[-1], 5)
        with self.assertRaises(ValueError):
            make_windows([1, np.nan], [0, 1], 2, 1)

    def test_fft_power_and_temporal_features(self):
        fs = 256
        x = np.sin(2 * np.pi * 10 * np.arange(512) / fs)[None, :]
        features = extract_features(x, fs).iloc[0]
        self.assertAlmostEqual(features.Alpha, .5)
        self.assertAlmostEqual(features.Energy, 256)
        self.assertAlmostEqual(features.RMS, np.sqrt(.5))
        self.assertAlmostEqual(features.Delta, 0)
        for n in [511, 512]:
            y = np.random.default_rng(2).normal(size=(1, n))
            _, power = fft_power(y, fs)
            self.assertAlmostEqual(power.sum() * fs / n, y.var())

    def test_clock_alignment_and_midnight(self):
        frame = pd.DataFrame({'start_s': [43551.], 'end_s': [43560.], 'annotation_unit': ['excel_days']})
        aligned = align_clock_annotations(frame, '12:00:00')
        self.assertEqual(aligned.start_s.iloc[0], 351)
        self.assertEqual(aligned.duration_s.iloc[0], 9)
        midnight = frame.assign(start_s=5., end_s=10.)
        self.assertEqual(align_clock_annotations(midnight, '23:59:55').start_s.iloc[0], 10)
        with self.assertRaises(ValueError):
            align_clock_annotations(frame, '25:00:00')

    def test_header_starts_and_excel_durations(self):
        events = pd.DataFrame({'onset_s': [0., 100., 200.],
                               'label': ['Début', 'ABSENCE', 'absence']})
        annotations = pd.DataFrame({'start_s': [34027., 34165.],
                                    'end_s': [34057., 34190.]})
        result = seizures_from_header(events, annotations)
        np.testing.assert_array_equal(result.start_s, [100., 200.])
        np.testing.assert_array_equal(result.end_s, [130., 225.])
        np.testing.assert_array_equal(result.duration_s, [30., 25.])
        with self.assertRaises(ValueError):
            seizures_from_header(events.iloc[:2], annotations)
        with self.assertRaises(ValueError):
            seizures_from_header(events.assign(label='Début'), annotations)

    def test_purge_shared_samples(self):
        meta = pd.DataFrame({'start_s': [0, 1, 2, 3, 4], 'end_s': [2, 3, 4, 5, 6]})
        np.testing.assert_array_equal(purge_overlaps(meta, [0, 1, 3, 4], [2]), [0, 4])

    def test_no_merge_across_unobserved_gap(self):
        meta = pd.DataFrame({'start_s': [5, 0, 1, 3], 'end_s': [6, 1, 2, 4]})
        detected = detected_intervals(meta, [1, 1, 1, 0])
        np.testing.assert_array_equal(detected[['start_s', 'end_s']], [[0, 2], [5, 6]])
        actual = pd.DataFrame({'start_s': [0.], 'end_s': [2.]})
        comparison = compare_intervals(detected, actual)
        self.assertEqual(comparison.IoU.tolist(), [1., 0.])
        self.assertTrue(detected_intervals(meta, [0]*4).empty)

    def test_four_models_end_to_end(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(100, 3)), columns=['a', 'b', 'c'])
        y = pd.Series((X.a > 0).astype(int))
        models, validation, best = train_models(X.iloc[:80], y.iloc[:80])
        scores, predictions, matrices = evaluate_models(models, X.iloc[80:], y.iloc[80:])
        self.assertEqual(len(scores), 4)
        self.assertIn(best, models)
        for matrix in matrices.values():
            self.assertEqual(matrix.shape, (2, 2))
            self.assertEqual(matrix.sum(), 20)
        self.assertTrue(np.isfinite(scores.to_numpy()).all())


if __name__ == '__main__':
    unittest.main()
