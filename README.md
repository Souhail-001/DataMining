# EEG data-mining project

The project covers every requirement through **section 19** of the assignment.

- [Executed notebook](notebooks/main.ipynb): code, explanations, graphs and results.
- [Report](reports/report.md): methodology, results, discussion and conclusion.
- [Detailed explanation](explanation.md): step-by-step reasoning, formulas, code and interpretation.

| Assignment | Where to find it |
| --- | --- |
| Questions 1–6 | Loading, exploration, annotations, sample labels and class balance |
| Questions 7–9 | Windows, temporal features and FFT band powers |
| Questions 10–11 | Class comparison and feature correlations |
| Questions 12–14 | Four models, evaluation and temporal detection |
| Section 15 | False positives, false negatives, true positives and true negatives |
| Section 16 | Proposed improvements and limitations |
| Section 18 | Notebook and report |
| Section 19 | Complete pipeline summary |

The PDF skips section 17.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/main.ipynb
```

Choose **Run All**. The notebook regenerates its figures, CSV tables and report.

## Data convention

The recording starts at **0 seconds**. For `210928B_B_0000`, two explicit `ABSENCE`
markers in the header provide elapsed seizure starts. Excel provides durations
(`FinCE − DebCE`, in HH:MM:SS). Pair them chronologically and calculate
`end = start + duration`. The pairing is an assumption; the notebook and report
explain the timing discrepancy between the two sources.

The baseline uses one EEG channel, 2-second windows and 50% overlap. After a
stratified 80/20 split, training windows sharing samples with test windows are
removed. Training-only validation selects the model. Results describe one recording.

## Project files

| Folder | Purpose |
| --- | --- |
| `data/raw/` | Original data, kept in one location |
| `notebooks/` | Main executed notebook |
| `src/` | Functions called by the notebook and their supporting loader code |
| `reports/` | Report, figures and result tables |
| `outputs/results/` | Sample labels, window features and test predictions |
| `tests/` | Checks for labels, features, splitting, models and intervals |

Run checks with `python -m unittest discover -s tests -v`.
