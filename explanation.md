# Understanding the EEG data-mining project

This document explains what the project does, why each step is needed, how the code works, and how to read the results. It follows the assignment through section 19. The PDF skips section 17.

The [notebook](notebooks/main.ipynb) contains the executable analysis. The [report](reports/report.md) summarizes the measured results. This explanation provides the reasoning behind them. The numbers below describe the current notebook run on `210928B_B_0000`; changing the recording or parameters requires rerunning the analysis and updating these explanations.

## 1. What problem are we solving?

An EEG records electrical activity over time through several channels. We want to identify time periods labeled as seizures.

This is a **supervised binary classification** problem:

- **Class 0:** no seizure according to the labels we constructed.
- **Class 1:** seizure according to those labels.

“Supervised” means that the model learns from examples whose classes are already known. The model receives numerical descriptions of short EEG windows, then predicts a class for each window.

The complete path is:

```text
EEG signal + annotations
          ↓
Inspect the data and construct sample labels
          ↓
Split the signal into windows
          ↓
Calculate features for each window
          ↓
Separate training and test data
          ↓
Train and compare four classifiers
          ↓
Predict classes for test windows
          ↓
Reconstruct predicted intervals and analyze errors
```

A sample, a window and a seizure are different things. A sample is one measurement at one instant. A window contains many consecutive samples. A seizure may extend across many windows. Our classification scores measure **windows**, not complete seizure episodes.

## 2. Which files do we use?

The assignment describes EEG stored in Excel, but the supplied signals are MATLAB files. We use the actual supplied format.

| File | Role |
| --- | --- |
| `data/raw/210928B_B_0000d.mat` | EEG measurements and elapsed sample times |
| `data/raw/210928B_B_0000h.mat` | Event markers, including two explicit `ABSENCE` markers |
| `data/raw/Crises absences.xlsx` | Annotated seizure clock intervals, from which durations are calculated |

We selected this recording because it contains two explicit seizure markers in its header and two seizure intervals in Excel. The previously explored recording, `230102B_F_0000`, did not contain explicit seizure markers. It is not the recording used for the final model results.

All supplied original recordings remain in `data/raw/`. The notebook analyzes **one recording and one EEG channel**, `EEGFp1`, to keep the exercise understandable. It does not train on all recordings.

## 3. Q1–Q2: loading and exploring the signal

### How the MATLAB loader works

The MATLAB files contain a `timetable`, not a plain numeric matrix. Its data are stored inside MATLAB's object workspace. `src/data.py` reads that workspace, finds the timetable, retrieves channel names and concatenates consecutive sample blocks.

The loader converts MATLAB duration values from milliseconds to seconds. It determines the sampling frequency from the number of samples per block and the time between blocks. It checks that block spacing is regular.

This technical decoding is kept in a module so the notebook can load the recording with one call:

```python
recording = load_recording(RECORDING_ID, ROOT)
```

The resulting table has a `Time` column and one column per measurement channel.

### What we found

| Property | Result |
| --- | ---: |
| Samples | 344,064 |
| Original variables, including `Time` | 22 |
| Sampling frequency | 256 Hz |
| Recording duration | 1,344 seconds, or 22 min 24 s |
| Missing values | 0 |

A sampling frequency of 256 Hz means **256 measurements per second**. The time between two samples is `1 / 256 = 0.00390625` seconds.

The recording begins at **0 seconds**. Its duration is `344064 / 256 = 1344` seconds; its last sample occurs one sampling period before that endpoint.

We display the first rows and plot EEG amplitude against elapsed time. Amplitude units are absent from the source metadata, so we label them **source units** rather than claiming they are microvolts.

For the long overview plot, only every 16th sample is drawn to keep rendering manageable. Feature extraction still uses every original sample. This display reduction can miss narrow spikes, so it is not a substitute for inspecting the detailed signal.

### Preprocessing actually performed

We check for missing or infinite values and select one EEG channel. The notebook stops if that channel contains invalid numeric values. It does not silently fill or remove samples.

No artifact filter, notch filter or bandpass filter is applied in the final baseline. Scaling for KNN and SVM happens later, inside their training pipelines. Mean removal for the FFT is also a separate operation described below.

## 4. Q3–Q4: constructing seizure intervals

### Two sources with different roles

Following the user's clarified convention:

1. Read each seizure's **elapsed start** from an explicit header marker.
2. Calculate its **duration** from Excel: `FinCE − DebCE`.
3. Pair the first header seizure with the first Excel interval, then the second with the second.
4. Calculate `elapsed end = elapsed start + duration`.

Excel clock times mean **HH:MM:SS**. Numeric Excel values store fractions of a day, so multiplying by 86,400 gives seconds. For example, the first Excel interval is 09:27:07–09:27:37, a duration of 30 seconds.

We do **not** use that clock time as an elapsed time. Because the header already provides elapsed seizure starts, no recording-start clock offset is needed.

### Final labeling intervals

| Seizure | Header start (s) | Excel duration (s) | Calculated end (s) |
| --- | ---: | ---: | ---: |
| 1 | 1162.1640625 | 30 | 1192.1640625 |
| 2 | 1291.9140625 | 25 | 1316.9140625 |

Mean duration: `(30 + 25) / 2 = 27.5` seconds. Minimum: 25 seconds. Maximum: 30 seconds.

The gap between the two constructed intervals is:

```text
1291.9140625 − 1192.1640625 = 99.75 seconds
```

This is the seizure-free gap, not the start-to-start difference. The header start-to-start difference is 129.75 seconds.

### The important assumption

`seizures_from_header()` requires equal numbers of explicit markers and Excel intervals, then pairs them chronologically. The sources do not contain shared seizure identifiers proving those pairs.

There is a timing discrepancy: Excel starts are 138 seconds apart, whereas header starts are 129.75 seconds apart. The difference is **8.25 seconds**. Annotation delay is one possible explanation, but we have not established its cause.

Therefore, these intervals implement the requested source convention; they are not independently verified exact seizure boundaries. We do not silently adjust them to make the two sources agree.

## 5. Q5–Q6: assigning a class to every sample

`label_samples()` creates a copy of the raw table and adds `Class`. A sample is positive if:

```text
seizure start ≤ sample time ≤ seizure end
```

Both endpoints are included, as requested by the assignment. All other samples receive class 0. The original measurements remain unchanged.

The function checks recording identity, increasing sample times, valid intervals and acquisition bounds. It rejects intervals that contain no sampled instant.

| Sample class | Count | Percentage |
| --- | ---: | ---: |
| 0 | 329,982 | 95.907% |
| 1 | 14,082 | 4.093% |

The classes are strongly imbalanced: about 96% of samples are negative. The positive sample count is slightly greater than `55 seconds × 256 Hz` because the labeling rule includes the aligned interval endpoints.

The labeled plot shows larger oscillations during parts of both intervals. However, a much larger isolated transient near 1242 seconds lies outside them. A large amplitude by itself does not establish a seizure.

The full labeled table exists in notebook memory. We export just `Time` and `Class` to `outputs/results/sample_labels.csv.gz` to avoid duplicating every measurement column on disk.

## 6. Q7: cutting the signal into windows

One isolated EEG sample contains little information about a changing signal. We summarize short sequences instead.

Our settings are:

- Window length: **2 seconds**, or 512 samples.
- Overlap: **50%**.
- Step between window starts: **1 second**, or 256 samples.

The first windows cover `[0, 2)`, `[1, 3)` and `[2, 4)` seconds. Window ends are exclusive: the first window contains samples from 0 through 1.99609375 seconds. This is distinct from the inclusive seizure-label boundaries.

A window is class 1 when **at least 50% of its samples** are positive. For 512 samples, that means at least 256 positive samples. A brief seizure overlap below that threshold still gives a class-0 window.

The number of complete windows is:

```text
floor((344064 − 512) / 256) + 1 = 1343
```

There are **1,288 negative windows and 55 positive windows**. The function discards an incomplete final window if one exists.

## 7. Q8: calculating temporal features

A feature is a number that describes one window. Instead of giving a model all 512 amplitudes, we give it a small set of summaries.

For samples `x₁, …, xₙ`, with `n = 512` and mean `μ`:

| Feature | Calculation | Meaning |
| --- | --- | --- |
| Mean | `sum(x) / n` | Average signal level |
| Variance | `sum((x − μ)²) / n` | Spread around the mean |
| Standard deviation | `sqrt(variance)` | Spread in amplitude units |
| Minimum | Smallest sample | Lowest amplitude |
| Maximum | Largest sample | Highest amplitude |
| Amplitude | `maximum − minimum` | Peak-to-peak range |
| RMS | `sqrt(sum(x²) / n)` | Overall magnitude, including any offset |
| Energy | `sum(x²)` | Sum of squared amplitudes |

The code uses population variance and standard deviation: divide by `n`, not `n − 1`. “Energy” here is the discrete sum of squares; it is not multiplied by the sampling period and is not a calibrated physical energy measurement.

Some features contain related information. `Variance = Std²`, and for these fixed-length windows, `Energy = 512 × RMS²`. Keeping all eight satisfies the assignment; later feature selection could remove redundancy.

## 8. Q9: FFT and frequency-band power

Temporal features describe amplitudes. Frequency features describe how much of the signal's variation lies at different oscillation frequencies.

The FFT transforms each window into frequency components. Our implementation:

1. Subtracts the window mean, removing its constant offset for this calculation only.
2. Calculates the real-valued FFT (`rfft`).
3. Squares its magnitude and divides by `sampling_rate × window_size` to obtain a power spectral density.
4. Doubles the positive-frequency contributions, except the DC and Nyquist endpoints, because the negative-frequency half is omitted.
5. Sums spectral density within each band and multiplies by the bin width.

The bin width is `256 / 512 = 0.5 Hz`. We do not zero-pad or apply a taper. A finite rectangular window can spread power across nearby frequencies; these are simple periodogram estimates.

| Band | Included frequencies |
| --- | --- |
| Delta | `0.5 ≤ f < 4 Hz` |
| Theta | `4 ≤ f < 8 Hz` |
| Alpha | `8 ≤ f < 13 Hz` |
| Beta | `13 ≤ f < 30 Hz` |

Using an exclusive upper bound prevents a frequency bin being counted twice. Band power has squared source-amplitude units. FFT mean removal does not change the temporal features calculated from the raw window.

The final dataset has **12 predictors**: eight temporal features and four band powers. It also includes `start_s`, `end_s`, `seizure_fraction` and `Class`. These four metadata/target columns are **not model inputs**. In particular, using `seizure_fraction` as a predictor would reveal the answer to the model.

## 9. Q10–Q11: understanding the features

We compare feature means between classes and draw a correlation matrix.

Examples from the current results:

| Feature | Class 0 mean | Class 1 mean |
| --- | ---: | ---: |
| Standard deviation | 49.370 | 64.931 |
| Peak-to-peak amplitude | 189.700 | 318.899 |
| Delta power | 2396.827 | 1373.938 |
| Theta power | 65.240 | 189.699 |
| Alpha power | 21.020 | 154.484 |
| Energy | 6,518,122 | 3,644,217 |

Not every feature is larger during seizures. For example, average energy is larger in class 0. Large non-seizure transients can affect these summaries; the table alone does not establish a cause.

Pearson correlation measures linear association, from −1 to +1. We list pairs with absolute correlation at least 0.9; there are eight such pairs in this run. Nonlinear redundancy, such as squaring a standard deviation, need not produce a correlation of exactly 1.

We also rank features using the absolute difference between class means divided by a pooled within-class standard deviation. The largest differences here are **Min, Amplitude and Alpha**.

These analyses use the full dataset for description. They are not used to select features or tune models, which would otherwise let information from the test set influence training. A large descriptive difference does not guarantee good prediction on new recordings.

## 10. Q12: separating training and test data

Training data teach the model. Test data estimate how it performs on examples it did not train on.

We first perform the required stratified 80/20 split with random seed 42. “Stratified” means approximately preserving the class proportions.

| Stage | Windows |
| --- | ---: |
| Initial training set | 1,074 |
| Test set | 269 |
| Retained training set after overlap removal | 694 |

### Why do we remove some training windows?

With 50% overlap, adjacent windows share samples. For example, training on `[0, 2)` and testing on `[1, 3)` would expose half the test signal during training.

`purge_overlaps()` removes every training window that shares time with a test window. Touching endpoints alone do not count as overlap because windows use exclusive end boundaries. This removes 380 initial training windows. The test set is unchanged.

Thus **80/20 describes the initial split**, not the retained sizes after removal. This precaution prevents shared-sample leakage but does not make windows from the same recording independent.

## 11. The four classifiers

| Model | Simple explanation | Settings in this project |
| --- | --- | --- |
| Decision Tree | Learns a sequence of feature-threshold questions | Maximum depth 5; balanced class weights |
| Random Forest | Combines predictions from many decision trees | 100 trees; maximum depth 8; balanced weights |
| KNN | Predicts from the classes of nearby training examples | Five neighbors; feature scaling |
| SVM | Learns a boundary separating classes in feature space | RBF kernel; `C=1`; balanced weights; scaling |

Balanced weights give the rarer class more influence during training. They do not create new positive examples or guarantee improved recall. KNN does not use those class weights here.

Scaling transforms a feature approximately as `(value − training mean) / training standard deviation`. It matters for KNN distances and the SVM boundary because energy and amplitude have very different numerical scales.

`StandardScaler` is inside the KNN/SVM pipelines. It is fitted only on each training fold and then applied to validation/test data. Scaling the entire dataset before splitting would leak information.

## 12. Selecting the model without using the test set

We use three-fold stratified cross-validation within the retained training set. For each fold, train on two parts and validate on the third. Remove overlapping training windows relative to that validation part too.

The average validation F1 chooses the model:

| Model | Mean training-validation F1 |
| --- | ---: |
| Decision Tree | 0.4843 |
| Random Forest | 0.5710 |
| KNN | **0.5954** |
| SVM | 0.4583 |

We therefore select **KNN**, then use its test predictions for temporal reconstruction. Every model is fitted on the full retained training set after validation, so their test scores can be compared.

These folds still contain windows from the same recording and potentially the same seizure episode. Overlap removal does not solve patient-level or episode-level dependence.

## 13. Evaluation and Q13: reading the metrics

A confusion matrix counts four outcomes:

| Outcome | Meaning |
| --- | --- |
| TP: true positive | Positive window correctly predicted positive |
| TN: true negative | Negative window correctly predicted negative |
| FP: false positive | Negative window incorrectly predicted positive |
| FN: false negative | Positive window incorrectly predicted negative |

Rows of our matrices are actual classes; columns are predictions: `[[TN, FP], [FN, TP]]`.

| Metric | Formula | Question it answers |
| --- | --- | --- |
| Accuracy | `(TP + TN) / total` | What fraction of all windows is correct? |
| Precision | `TP / (TP + FP)` | Of positive predictions, how many are correct? |
| Recall | `TP / (TP + FN)` | Of actual positive windows, how many were found? |
| F1 | `2 × precision × recall / (precision + recall)` | How well are precision and recall balanced? |

Undefined positive-class metrics are reported as zero rather than causing an error.

### Actual test results

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.9554 | 0.4444 | 0.3636 | 0.4000 |
| Random Forest | 0.9665 | 0.5714 | 0.7273 | 0.6400 |
| KNN | 0.9740 | 0.7500 | 0.5455 | 0.6316 |
| SVM | 0.9442 | 0.4091 | 0.8182 | 0.5455 |

The Random Forest has a slightly higher **test** F1 than KNN, but KNN remains the selected model because selection was based on **training-validation** F1. Choosing again after seeing the test scores would use the test set for model selection.

An always-negative classifier gets **95.91% accuracy** on this test set while detecting no positive windows. This explains why accuracy alone is misleading. Recall matters for missed seizures, precision for unnecessary alerts, and F1 balances both. SVM has the highest recall here, but also more false alerts.

## 14. Q14: converting predictions into time intervals

For each test window, we retain its start, end and predicted class. `detected_intervals()` sorts positive windows and merges their overlapping or touching time spans.

For example, positive windows `[10, 12)` and `[11, 13)` become `[10, 13)`. Positive windows `[10, 12)` and `[15, 17)` remain separate because there is an uncovered gap.

This is a union of positive window spans, not a new prediction for every individual sample. If positive and negative windows overlap, the interval union does not perform majority voting across them.

The selected KNN produces six intervals:

| Predicted start (s) | Predicted end (s) | Temporal IoU with closest-overlapping annotation |
| --- | ---: | ---: |
| 1160 | 1162 | 0.0000 |
| 1163 | 1165 | 0.0667 |
| 1168 | 1170 | 0.0667 |
| 1171 | 1173 | 0.0667 |
| 1290 | 1292 | 0.0032 |
| 1303 | 1308 | 0.2000 |

IoU means **intersection over union**: overlapping duration divided by the total duration covered by either interval. For each prediction, we select the annotation giving the largest IoU. This is not one-to-one event matching; several predicted fragments may correspond to the same annotated seizure.

The test windows are scattered across time. Untested periods are **unknown**, not negative. The six intervals are therefore fragments from test predictions, not proof that six complete seizures occurred. We do not predict on training windows and present those predictions as test performance.

## 15. Error analysis

For KNN, the test confusion matrix is:

```text
                  Predicted 0    Predicted 1
Actual 0              256              2
Actual 1                5              6
```

This gives:

- Precision: `6 / (6 + 2) = 0.75`.
- Recall: `6 / (6 + 5) ≈ 0.5455`.
- F1: `12 / (12 + 2 + 5) ≈ 0.6316`.
- Accuracy: `(256 + 6) / 269 ≈ 0.9740`.

Only 11 test windows are positive, so one changed prediction would move recall by about 9.09 percentage points. The score is consequently sensitive to this small test set.

False positives create unnecessary alerts. False negatives miss positive windows. Their possible causes include artifacts, boundary effects, imperfect labels and similar features in both classes. The counts alone do not tell us which cause applies.

Every test window and its outcome is saved in `outputs/results/test_predictions.csv`. This allows inspection of errors at their actual times rather than relying only on aggregate scores.

## 16. Improvements and limitations

The assignment asks us to propose improvements. We propose these without claiming they have already improved the results:

1. **Verify the annotation pairing.** Resolve the timing discrepancy between Excel and header markers; better labels are necessary for trustworthy evaluation.
2. **Evaluate on separate recordings or patients.** This tests whether the model generalizes beyond one recording. Splitting by seizure episode could also reduce dependence between sets.
3. **Compare window sizes and overlap.** Shorter windows may localize boundaries better; longer windows provide more signal context. Compare choices using training-only validation.
4. **Test artifact handling and additional channels.** Filtering or multiple channels may help distinguish seizures from unrelated changes. Their benefit must be measured.
5. **Tune class weights or decision thresholds.** Use validation data to choose a suitable tradeoff between missed windows and false alerts.

Scaling and balanced weights are already present where described. They should not be presented as future changes that were never implemented.

This remains an educational, single-recording analysis with two labeled episodes and an assumed marker correspondence. It does not establish performance on unseen patients or clinical reliability.

## 18–19. Deliverables and how the code fits together

| File | Responsibility |
| --- | --- |
| `notebooks/main.ipynb` | Runs all steps in order, displays results and writes the report |
| `src/data.py` | Decodes MATLAB data and reads Excel annotations |
| `src/labeling.py` | Selects annotations, constructs intervals, calculates statistics and labels samples |
| `src/windowing.py` | Creates complete windows and their classes |
| `src/features.py` | Calculates temporal features and FFT powers |
| `src/training.py` | Removes overlaps, trains four models and selects by validation F1 |
| `src/evaluation.py` | Calculates test metrics and confusion matrices |
| `src/detection.py` | Reconstructs positive intervals and compares temporal overlaps |
| `reports/report.md` | Concise report generated from the notebook results |
| `tests/` | Checks important calculations and model behavior |

The final section of the notebook explicitly shows the full pipeline required by section 19. The notebook and report satisfy section 18.

### Where to find the outputs

| Location | Contents |
| --- | --- |
| `reports/figures/` | Signal, labels, FFT examples, class comparisons, correlations, confusion matrices and temporal detections |
| `reports/tables/` | Clock intervals, constructed intervals, statistics, validation scores, test metrics and interval comparisons |
| `outputs/results/sample_labels.csv.gz` | Sample times and binary labels |
| `outputs/results/window_features.csv` | Window metadata, target and 12 features |
| `outputs/results/test_predictions.csv` | Test-window predictions and TP/TN/FP/FN outcomes |

The final notebook cell regenerates `reports/report.md`; manual edits to that report will be overwritten on the next run. This explanation file is maintained separately.

### How to reproduce the analysis

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/main.ipynb
```

Choose **Run All**. The notebook finds the project root from either the root folder or `notebooks/`. No clock offset needs to be entered.

To run the checks:

```bash
python -m unittest discover -s tests -v
```

The completed real-data notebook ran successfully, and all **13 tests passed**. The tests cover time conversion, annotation validation, sample boundaries, header starts plus Excel durations, window thresholds, FFT normalization, overlap removal, interval reconstruction and the four-model pipeline. Synthetic test cases check the implementation; the reported model scores come from the actual recording.

Random seed 42 makes the splits and randomized models repeatable within the same software environment. `requirements.txt` lists dependencies without exact version pins, so future library versions may produce small differences.

### What was removed during cleanup?

We removed a byte-verified duplicate raw-data folder, the unused old `venv`, unused preprocessing/visualization modules, obsolete figures and labels from the earlier recording, notebook checkpoints, caches and empty directories.

The original data remain in `data/raw/`. The working `.venv`, the executed notebook, source code, tests and current results are retained. Git and environment-managed configuration folders were left intact.
