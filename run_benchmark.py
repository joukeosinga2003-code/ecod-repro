"""Reproduce the main experiment of the ECOD paper (Tables 4 and 5).

What this script does, for each of the paper's 14 ODDS datasets:
  1. load the data (and, for Shuttle, take 10,000 random rows like the paper)
  2. split it randomly into 60% training and 40% test data
  3. fit ECOD and 9 baseline detectors on the training part, score the test part
  4. measure how well the scores separate outliers from normal points (ROC and AP)
Steps 2-4 are repeated 10 times with a different random split, and averaged.

ECOD itself is NOT implemented here: we use the authors' own implementation, which is
part of the PyOD library (the paper links to it). The same goes for the baselines.

Usage:
  uv run python run_benchmark.py                          # full experiment
  uv run python run_benchmark.py --datasets wbc --trials 1  # quick test
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

warnings.filterwarnings("ignore")  # some baselines print many harmless warnings

# ---------------------------------------------------------------------------------
# Settings. These are the "parameters" of the experiment (see README, Parameters).
# ---------------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
TEST_SIZE = 0.4  # paper §4.1: 60% training, 40% test
N_TRIALS = 10  # paper §4.1: average over 10 independent trials
SHUTTLE_ROWS = 10_000  # paper §4.1: Shuttle is cut down to 10,000 random rows
SHUTTLE_SEED = 0  # our choice: the paper does not publish which rows it used

# The 14 ODDS datasets from the paper's Table 1.
PAPER_DATASETS = [
    "arrhythmia", "breastw", "cardio", "ionosphere", "lympho", "mammography",
    "optdigits", "pima", "satellite", "satimage-2", "shuttle", "speech", "wbc", "wine",
]


def make_detectors(contamination, seed):
    """Create a fresh copy of every detector for one trial.

    All settings are PyOD's defaults, because the paper does not give any.
    `contamination` (the share of outliers) only decides which points get the label
    "outlier"; it does not change the scores, so it has no effect on ROC or AP.
    Detectors that use randomness get the trial number as seed, so runs are repeatable.
    """
    return {
        "ECOD": ECOD(contamination=contamination),  # the method of the paper
        "ABOD": ABOD(contamination=contamination),
        "CBLOF": CBLOF(contamination=contamination, random_state=seed),
        "HBOS": HBOS(contamination=contamination),
        "IForest": IForest(contamination=contamination, random_state=seed),
        "KNN": KNN(contamination=contamination),
        "LODA": LODA(contamination=contamination),
        "LOF": LOF(contamination=contamination),
        "OCSVM": OCSVM(contamination=contamination),
        "PCA": PCA(contamination=contamination, random_state=seed),
    }


# ---------------------------------------------------------------------------------
# Step 1: load a dataset
# ---------------------------------------------------------------------------------
def load_dataset(name):
    """Return X (one row per data point) and y (1 = outlier, 0 = normal).

    Most datasets are ODDS .mat files in data/. Four of them (breastw, mammography,
    speech, wine) are the same ODDS data saved as .npz files by ADBench, in data/adbench/.
    """
    mat_file = DATA_DIR / f"{name}.mat"
    if mat_file.exists():
        data = loadmat(mat_file)
    else:
        npz_file = next((DATA_DIR / "adbench").glob(f"*_{name}.npz"))
        data = np.load(npz_file, allow_pickle=False)
    X = data["X"].astype(float)
    y = data["y"].ravel().astype(int)

    if name == "shuttle":
        rng = np.random.default_rng(SHUTTLE_SEED)
        rows = rng.choice(len(X), SHUTTLE_ROWS, replace=False)
        X, y = X[rows], y[rows]
    return X, y


# ---------------------------------------------------------------------------------
# Steps 2-4: split, fit, score; repeated N_TRIALS times
# ---------------------------------------------------------------------------------
def run_dataset(name, n_trials):
    X, y = load_dataset(name)
    contamination = y.mean()  # share of outliers in this dataset
    results = []

    for trial in range(n_trials):
        # Step 2: random 60/40 split. The seed is the trial number (0, 1, ..., 9), so
        # every detector sees the same 10 splits. `stratify=y` keeps the share of
        # outliers equal in both parts (our choice; the paper does not say).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=trial, stratify=y
        )

        for detector_name, detector in make_detectors(contamination, trial).items():
            # LODA has no seed option and uses numpy's shared random generator, so we
            # seed that too; otherwise LODA gives different results on every run.
            np.random.seed(trial)
            start = time.perf_counter()
            try:
                # Step 3: learn from the training data (labels are NOT used: the
                # methods are unsupervised), then give every test point a score.
                # A higher score means "more likely an outlier".
                detector.fit(X_train)
                scores = detector.decision_function(X_test)

                # Step 4: compare the scores with the true labels.
                # ROC: chance that a random outlier scores higher than a random normal
                #      point (1.0 = perfect, 0.5 = random guessing).
                # AP:  average precision when going down the ranking from the highest
                #      score; rewards putting the outliers at the very top.
                roc = roc_auc_score(y_test, scores)
                ap = average_precision_score(y_test, scores)
            except Exception as error:  # e.g. ABOD fails on breastw; keep going
                print(f"  {name}/{detector_name} failed: {error}")
                roc = ap = np.nan

            results.append({
                "dataset": name, "trial": trial, "detector": detector_name,
                "roc_auc": roc, "ap": ap, "seconds": time.perf_counter() - start,
            })
    return results


# ---------------------------------------------------------------------------------
# Run everything and save the results
# ---------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=PAPER_DATASETS,
                        help="dataset names (default: the paper's 14 ODDS datasets)")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--out", default="results", help="folder for the result files")
    args = parser.parse_args()

    all_results = []
    for name in args.datasets:
        print(f"== {name}", flush=True)
        all_results += run_dataset(name, args.trials)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(out / "raw.csv", index=False)  # one line per dataset/trial/detector

    # Average over the trials: one table per measure, datasets x detectors
    for measure in ("roc_auc", "ap", "seconds"):
        table = df.pivot_table(index="dataset", columns="detector", values=measure, aggfunc="mean")
        table.loc["MEAN"] = table.mean()
        table.round(4).to_csv(out / f"{measure}.csv")
        print(f"\n{measure} (mean over {args.trials} trials)")
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
