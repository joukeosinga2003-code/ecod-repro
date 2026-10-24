# ECOD reproduction

Reproduction of the main experiment (Tables 4 and 5) of

> Z. Li, Y. Zhao, X. Hu, N. Botta, C. Ionescu, G. H. Chen. *ECOD: Unsupervised Outlier
> Detection Using Empirical Cumulative Distribution Functions.* IEEE TKDE, 2022.
> DOI: 10.1109/TKDE.2022.3159580

ECOD is compared with 9 baseline detectors on the 14 ODDS datasets of the paper, using
ROC-AUC and average precision (AP), averaged over 10 random 60/40 train/test splits.

## How to run

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.14 and all dependencies).

```bash
uv sync                                  # create .venv from uv.lock
uv run python run_benchmark.py           # full run: 14 datasets x 11 detectors x 10 trials (~10-20 min)
uv run python compare.py                 # our results next to the paper's Tables 4 and 5
uv run python check_variants.py          # which ECOD variant (if any) matches the paper
```

Quick test on one dataset: `uv run python run_benchmark.py --datasets wbc --trials 1`.

Outputs go to `results/`: `raw.csv` (every dataset/trial/detector), `roc_auc.csv`,
`ap.csv`, `seconds.csv` (means per dataset), and `compare_*.csv`.

## Files

| File | Purpose |
|---|---|
| `run_benchmark.py` | Runs the experiment. All protocol settings are at the top of the file. |
| `src/ecod_repro/ecod_paper.py` | ECOD implemented exactly as Algorithm 1 of the paper. |
| `compare.py` | Compares `results/raw.csv` with the paper's numbers. |
| `check_variants.py` | Tests alternative ECOD variants against the paper's per-dataset numbers. |
| `data/*.mat` | ODDS datasets ([odds.cs.stonybrook.edu](http://odds.cs.stonybrook.edu)). |
| `data/adbench/*.npz` | breastw, mammography, speech, wine: ODDS data via the [ADBench](https://github.com/Minqi824/ADBench) mirror (shapes match the paper's Table 1). |
| `data/paper_tables_4_5.csv` | The paper's Table 4 (ROC) and Table 5 (AP) values for the 14 ODDS datasets. |

`data/` also holds 7 ODDS datasets that the paper does not use (glass, letter, mnist,
musk, pendigits, vertebral, vowels); they are only run when named with `--datasets`.

## Parameters

| Parameter | Value | Source |
|---|---|---|
| Train/test split | 60% / 40%, stratified, `random_state` = trial number (0-9) | paper §4.1 (stratification and seeds are our choice) |
| Trials | 10, scores averaged | paper §4.1 |
| Metrics | ROC-AUC, average precision (scikit-learn) | paper §4.1 |
| Datasets | 14 ODDS datasets of Table 1 | paper Table 1 |
| Shuttle | 10,000 random rows, seed 0 | paper §4.1 (seed is our choice) |
| Scaling | none | not mentioned in the paper |
| `contamination` | true outlier fraction of the dataset | only affects labels, not scores/metrics |
| Detector settings | PyOD 3.6.7 defaults; `random_state` = trial for IForest, PCA, CBLOF | paper gives no settings |
| ECOD | no hyperparameters | paper |

## Results

Mean over the 14 ODDS datasets:

| Detector | ROC paper | ROC ours | AP paper | AP ours |
|---|---|---|---|---|
| ECOD (PyOD) | **0.847** | 0.799 | **0.608** | 0.524 |
| ECOD (Algorithm 1) | – | 0.798 | – | 0.479 |
| IForest | 0.836 | **0.838** | 0.574 | 0.560 |
| CBLOF | 0.763 | **0.838** | 0.466 | 0.555 |
| PCA | 0.804 | 0.828 | 0.534 | **0.573** |
| HBOS | 0.832 | 0.817 | 0.543 | 0.518 |
| KNN | 0.757 | 0.795 | 0.424 | 0.470 |
| OCSVM | 0.797 | 0.762 | 0.528 | 0.399 |
| ABOD* | 0.663 | 0.744 | 0.291 | 0.359 |
| LODA | 0.737 | 0.736 | 0.455 | 0.453 |
| LOF | 0.666 | 0.696 | 0.327 | 0.388 |

\* ABOD returns NaN on breastw (duplicate rows), so its mean covers 13 datasets.
LSCP and SUOD (ensembles) from the paper are not included.

The baselines are close to the paper (IForest +0.003, LODA -0.001, HBOS -0.015 ROC),
but ECOD scores about 0.05 ROC lower and ranks 5th instead of 1st. Per-dataset
numbers are in `results/compare_*.csv`.

## Findings

1. **PyOD's ECOD differs from the paper.** PyOD takes the maximum of the left, right and
   skewness-selected tail score *per dimension* and then sums. The skewness-selected score
   always equals one of the other two, so the skewness step has no effect. The paper
   (Algorithm 1, Eq. 6) sums each of the three over all dimensions first and takes the
   maximum of the three totals. `ecod_paper.py` implements the paper's version.
2. **Neither version reproduces the paper's ECOD numbers.** `check_variants.py` also tests
   right-tail-only scoring and ECDFs estimated on the test split alone. None closes
   the gap; the biggest differences are on ionosphere, pima, satellite, satimage-2 and
   wine.
3. **The paper contains inconsistencies.** In Table 1 the DAMI (`.arff`) rows are shifted
   (e.g. "Ionosphere (arff)" has Pima's 768 x 8). Table 2 reports values that cannot be
   right (ECOD-B scores 1.000 ROC on Pima, where the best detector in Table 4 reaches
   0.708), and the text and table disagree on the ECOD-L average. For this reason the 16
   DAMI datasets are not used here.
4. **Unknown details.** The paper does not give the PyOD version, random seeds, scaling,
   or detector settings, so all of these are assumptions here (see Parameters).
