# ECOD reproduction

This repository reproduces the main experiment (Tables 4 and 5) of:

> Z. Li, Y. Zhao, X. Hu, N. Botta, C. Ionescu, G. H. Chen. *ECOD: Unsupervised Outlier
> Detection Using Empirical Cumulative Distribution Functions.* IEEE TKDE, 2022.
> DOI: 10.1109/TKDE.2022.3159580

The paper claims that ECOD finds outliers better than 11 other methods. We check this
by running ECOD and 9 of those methods on the same public datasets, measuring them the
same way, and comparing our numbers with the paper's.

**ECOD is not reimplemented here.** The authors published their code as part of the
PyOD library, and the paper links to it. We use that code (`from pyod.models.ecod import
ECOD`). The baselines also come from PyOD. What this repository adds is the experiment
around them: loading the data, splitting it, running the methods and scoring them.

## How the experiment works

For each of the 14 datasets, `run_benchmark.py` does this:

1. **Load the data.** Each dataset is a table of data points plus a label that says
   which points are real outliers. Shuttle is cut down to 10,000 random rows, as in
   the paper.
2. **Split it.** 60% of the rows go into a training part and 40% into a test part,
   picked at random.
3. **Run each method.** The method learns from the training part *without* seeing the
   labels (that is what "unsupervised" means), then gives every test point an outlier
   score.
4. **Score the method.** The labels are used only now, to check whether the real
   outliers got high scores:
   - **ROC** (ROC-AUC): the chance that a random outlier gets a higher score than a
     random normal point. 1.0 is perfect, 0.5 is guessing.
   - **AP** (average precision): how well the real outliers end up at the very top of
     the ranking.

Steps 2-4 are repeated 10 times with a different random split (seeds 0-9), and the
10 results are averaged. This follows section 4.1 of the paper, which says: 60/40
split, 10 independent trials, ROC and AP.

## How to run

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.14 and all packages.

```bash
uv sync                                                # install (first time only)
uv run python run_benchmark.py                         # full experiment, ~10-20 minutes
uv run python run_benchmark.py --datasets wbc --trials 1   # quick test, a few seconds
uv run python compare.py                               # our results next to the paper's
```

All random choices use fixed seeds, so running it again gives the same numbers.

## Files

| File | What it is |
|---|---|
| `run_benchmark.py` | The experiment (steps 1-4 above). All settings are at the top. |
| `compare.py` | Puts our results next to the paper's Tables 4 and 5. |
| `data/*.mat` | 10 datasets from the [ODDS repository](http://odds.cs.stonybrook.edu), the source the paper uses. |
| `data/adbench/*.npz` | The other 4 ODDS datasets (breastw, mammography, speech, wine), downloaded from the [ADBench](https://github.com/Minqi824/ADBench) copy of ODDS. |
| `data/paper_tables_4_5.csv` | The paper's numbers from Tables 4 and 5, for comparison. |
| `results/raw.csv` | Every single result: one line per dataset, trial and method. |
| `results/roc_auc.csv`, `ap.csv`, `seconds.csv` | Averages over the 10 trials. |
| `results/compare_*.csv` | Output of `compare.py`. |

All 14 datasets have exactly the number of rows, columns and outliers listed in the
paper's Table 1 (Shuttle after cutting it to 10,000 rows).

## Parameters

| Setting | Value | Where it comes from |
|---|---|---|
| Train/test split | 60% / 40% | paper §4.1 |
| Number of trials | 10, averaged | paper §4.1 |
| Measures | ROC-AUC and average precision (scikit-learn) | paper §4.1 |
| Datasets | the 14 ODDS datasets of Table 1 | paper Table 1 |
| Shuttle | 10,000 random rows | paper §4.1 |
| Split seeds | 0-9 (one per trial) | our choice, not in the paper |
| Stratified split | yes (same share of outliers in both parts) | our choice, not in the paper |
| Shuttle seed | 0 | our choice, not in the paper |
| Feature scaling | none | not mentioned in the paper |
| Method settings | PyOD 3.6.7 defaults; seed = trial number for IForest, PCA, CBLOF | paper gives none |
| `contamination` | the real share of outliers | has no effect on ROC/AP |
| ECOD settings | none (ECOD has no parameters) | paper |

## Results

Average over the 14 datasets:

| Method | ROC paper | ROC ours | AP paper | AP ours |
|---|---|---|---|---|
| ECOD | **0.847** | 0.799 | **0.608** | 0.524 |
| IForest | 0.836 | **0.838** | 0.574 | 0.560 |
| CBLOF | 0.763 | **0.838** | 0.466 | 0.555 |
| PCA | 0.804 | 0.828 | 0.534 | **0.573** |
| HBOS | 0.832 | 0.817 | 0.543 | 0.518 |
| KNN | 0.757 | 0.795 | 0.424 | 0.470 |
| OCSVM | 0.797 | 0.762 | 0.528 | 0.399 |
| ABOD* | 0.663 | 0.744 | 0.291 | 0.359 |
| LODA | 0.737 | 0.736 | 0.455 | 0.449 |
| LOF | 0.666 | 0.696 | 0.327 | 0.388 |

\* ABOD crashes on breastw (that dataset has duplicate rows), so its average is over 13
datasets. The paper's two ensemble methods, LSCP and SUOD, are not included.

Several baselines come out close to the paper (IForest +0.003, LODA -0.001, HBOS -0.015
ROC), which suggests the data and setup match. ECOD, however, scores about 0.05 ROC
lower than reported and ranks 5th instead of 1st. Per-dataset numbers are in
`results/compare_*.csv`.

## Possible reasons for the difference

1. **The released code does not do exactly what the paper describes.** The paper
   (Algorithm 1, Eq. 6) adds up the scores of all dimensions three times (left tail,
   right tail, and the tail chosen by skewness) and takes the largest of the three
   totals. PyOD's code instead picks the largest score *per dimension* and then adds
   those up. Because the skewness-chosen score is always equal to either the left or
   the right score, the skewness step then has no effect.
2. **The code may have changed since 2022.** We use PyOD 3.6.7; the paper does not say
   which version produced its numbers.
3. **The paper has errors in its tables.** In Table 1 the DAMI rows are shifted (e.g.
   "Ionosphere (arff)" has Pima's size, 768 x 8). That is why the 16 DAMI datasets are
   not used here. Table 2 also has values that cannot be right (1.000 ROC on Pima,
   where the best method in Table 4 gets 0.708).
4. **Details the paper leaves out.** It does not publish its experiment script, its
   splits, its Shuttle sample, its seeds, or the settings of the other methods, so these
   are our own choices (see Parameters).
