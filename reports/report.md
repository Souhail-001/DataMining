# EEG seizure detection — project report

## Introduction
Detect annotated seizure periods from EEG using a simple classification pipeline.
Recording: `210928B_B_0000`. Channel: `EEGFp1`. The scope is one recording.

## Data and annotations
There are 344,064 samples, 22 original variables,
256 Hz sampling, 1344 s duration and 0 missing values.
Amplitude units are not supplied. Excel annotation times are HH:MM:SS;
only their differences are used as durations. Following the requested convention,
explicit header seizure markers supply elapsed starts. The 2 markers
are paired chronologically with the 2 Excel intervals.
This pairing is assumed from order and count, not verified by shared event identifiers.
Excel onset spacing is 138 s; header onset spacing is 129.75 s (difference 8.25 s); header markers may have annotation delay.
These are operational labels for this exercise, not precisely validated clinical boundaries.

| seizure | start_s | end_s | duration_s |
| --- | --- | --- | --- |
| 1 | 1162.1641 | 1192.1641 | 30.0 |
| 2 | 1291.9141 | 1316.9141 | 25.0 |
There are 2 seizures. Durations range from 25.000
to 30.000 s, with a mean of 27.500 s.
The mean gap from one seizure end to the next start is 99.750 s.
Durations therefore vary; see the full annotation table in the notebook.

| Class | Samples | Percent |
| --- | --- | --- |
| 0 | 329982 | 95.907 |
| 1 | 14082 | 4.093 |

## Methodology
Closed annotation intervals label original samples. Complete 2-second windows,
50% overlap, and the 50% positive-sample rule produce 1343 windows,
including 55 positive windows. Eight temporal features and four FFT band
powers are extracted from raw EEG. FFT resolution is 0.5 Hz.
A stratified 80/20 split is followed by removal of overlapping training windows:
1074 initial train, 694 retained train and 269 test windows.
KNN/SVM scaling is fitted inside training folds. Training-only, overlap-purged 3-fold F1 selects
**KNN**; the test set is not used for selection.

## Exploration
In the labeled signal plot, oscillations are larger during parts of both labeled
intervals, but the largest isolated transient (near 1242 s) occurs outside them.
Amplitude alone is therefore insufficient. Only 4.09% of samples are positive.

The largest descriptive standardized class differences are: Min, Amplitude, Alpha.
There are 8 feature pairs with absolute correlation >= 0.9.
Standard deviation/variance and RMS/energy are mathematically redundant transformations.
Class means and correlation plots are available in the notebook; descriptive differences
are not evidence of performance on new patients.

## Results
| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| Decision Tree | 0.9554 | 0.4444 | 0.3636 | 0.4 |
| Random Forest | 0.9665 | 0.5714 | 0.7273 | 0.64 |
| KNN | 0.974 | 0.75 | 0.5455 | 0.6316 |
| SVM | 0.9442 | 0.4091 | 0.8182 | 0.5455 |

Always-negative baseline accuracy: 0.9591; baseline positive recall/F1: 0.
Recall matters for missed seizures; precision measures false alerts. F1 balances both.
Accuracy alone can conceal the class imbalance.

## 14–15 — Temporal detection and errors
The selected model produces 6 merged positive test intervals.
See `tables/q14_detected_intervals.csv` and `tables/q14_interval_comparison.csv` for boundaries
and comparisons with actual annotations. Untested gaps remain unknown; detections from this
scattered test set are fragments, not complete reconstructed seizures.

| Outcome | Windows |
| --- | --- |
| TN | 256 |
| FP | 2 |
| FN | 5 |
| TP | 6 |

False positives are unnecessary alerts; false negatives are missed positive windows.
Artifacts and boundary effects are possible causes, not established diagnoses.
Every test-window prediction and its outcome are saved in `../outputs/results/test_predictions.csv`.

## 16 — Conclusion and improvements
The complete pipeline runs from raw MATLAB EEG and Excel annotations to features,
four classifiers, evaluation, temporal detections and error analysis.
The results describe this recording only. Shared-patient correlation remains despite
purging shared samples. Validate on independent recordings, compare window durations,
and test artifact removal, additional channels or class-weight/threshold tuning using
training-only validation. These improvements are proposed, not claimed as measured gains.

## 18 — Deliverables
The executed notebook is `../notebooks/main.ipynb`; this file is the report.
The notebook contains every requested step, including plots and numerical results.
The PDF skips section 17.

## 19 — Complete pipeline
Raw EEG + Excel annotations → exploration and finite-value checks → sample labels
→ 2-second windows → eight temporal features + four FFT band powers
→ classification dataset → stratified split and overlap removal
→ Decision Tree / Random Forest / KNN / SVM → evaluation
→ model selected by training-validation F1 → predicted seizure starts and ends.

Preprocessing is deliberately minimal: inspect missing/nonfinite values, choose one
EEG channel, and fit scaling inside KNN/SVM training folds. No artifact filtering
is applied in this baseline. The notebook displays the actual window-level results;
the limitations of chronological annotation pairing and same-recording evaluation remain.
