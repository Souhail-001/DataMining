"""Focused checks of time conversion, interval arithmetic and sample boundaries."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import Recording, load_seizure_annotations, annotation_time_seconds
from src.labeling import label_samples, seizure_statistics, select_seizures


class AnnotationTests(unittest.TestCase):
    def intervals(self, starts=(0.5, 1.5), ends=(1., 2.)):
        return pd.DataFrame({'start_s': starts, 'end_s': ends})

    def recording(self):
        return Recording('TEST_A_0000', pd.DataFrame({
            'Time': np.arange(12) / 4, 'EEG': np.arange(12, dtype=float)
        }), 4.)

    def test_minutes_seconds_centiseconds(self):
        convert = lambda v: annotation_time_seconds(v, 'minutes_seconds_centiseconds')
        self.assertEqual(convert('11:19:13'), 679.13)
        self.assertEqual(convert((11 * 3600 + 19 * 60 + 13) / 86400), 679.13)
        self.assertEqual(convert(.5040625), 725.51)
        self.assertEqual(convert('11:59:99'), 719.99)
        self.assertEqual(convert('12:00:00'), 720.)
        for invalid in ['11:60:00', '11:19:100', '11:19', float('nan'), -1]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                convert(invalid)

    def test_closed_boundaries_and_raw_data_preserved(self):
        raw = self.recording()
        result = label_samples(raw, self.intervals())
        np.testing.assert_array_equal(result.Class, [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
        pd.testing.assert_frame_equal(result.drop(columns='Class'), raw.data)
        self.assertNotIn('Class', raw.data)

    def test_statistics_sorted_and_single_seizure(self):
        stats = seizure_statistics(self.intervals((1.5, .5), (2., 1.))).iloc[0]
        self.assertEqual(stats.mean_duration_s, .5)
        self.assertEqual(stats.mean_gap_s, .5)
        self.assertEqual(stats.mean_onset_interval_s, 1.)
        self.assertTrue(np.isnan(seizure_statistics(self.intervals((0.,), (1.,))).iloc[0].mean_gap_s))

    def test_invalid_intervals_rejected(self):
        for starts, ends in [((1.,), (.5,)), ((np.nan,), (1.,)), ((-1.,), (1.,)), ((2.,), (4.,)), ((.01,), (.02,))]:
            with self.subTest(starts=starts, ends=ends), self.assertRaises(ValueError):
                label_samples(self.recording(), self.intervals(starts, ends))
        with self.assertRaises(ValueError):
            label_samples(self.recording(), self.intervals((), ()))
        with self.assertRaises(ValueError):
            seizure_statistics(self.intervals((0., .5), (1., 2.)))

    def test_overlapping_labels_form_union(self):
        result = label_samples(self.recording(), self.intervals((.5, .75), (1., 1.5)))
        self.assertEqual(result.Class.sum(), 5)

    def test_excel_units_typos_missing_pairs_and_count(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'data/raw'
            raw.mkdir(parents=True)
            path = raw / 'Crises absences.xlsx'
            frame = pd.DataFrame({'ID': [' TEST-A '], 'NombreCE': [2],
                'DebutCE5': [.5], 'FinCE5': [.6], 'DebtCE11': [1.], 'FinCE11': [1.2]})
            frame.to_excel(path, index=False)
            annotations = load_seizure_annotations(folder, time_unit='minutes')
            selected = select_seizures(annotations, 'TEST_A_0000')
            np.testing.assert_allclose(selected.start_s, [30., 60.])
            np.testing.assert_allclose(selected.duration_s, [6., 12.])
            for unit, factor in [('seconds', 1), ('excel_days', 86400)]:
                self.assertEqual(load_seizure_annotations(folder, time_unit=unit).start_s.iloc[0], .5 * factor)
            with self.assertRaises(ValueError):
                select_seizures(annotations.assign(reported_count=3), 'TEST_A_0000')
            with self.assertRaises(ValueError):
                select_seizures(annotations, 'OTHER_A_0000')
            with self.assertRaises(ValueError):
                label_samples(self.recording(), annotations.assign(recording_id='OTHER-A'))
            frame.loc[0, 'FinCE5'] = np.nan
            frame.to_excel(path, index=False)
            with self.assertRaises(ValueError):
                load_seizure_annotations(folder, time_unit='minutes')


if __name__ == '__main__':
    unittest.main()
