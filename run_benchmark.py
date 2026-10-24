"""Reproduce the ECOD vs. baselines comparison (ROC-AUC, average precision) on ODDS datasets.

Protocol knobs (TEST_SIZE, N_TRIALS, STANDARDIZE, per-detector params) are set at the
top so the report's parameter table can be read straight from this file. Verify them
against the paper's experimental setup section before reporting.
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
from sklearn.preprocessing import StandardScaler

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

from ecod_repro.ecod_paper import ECODPaper

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
TEST_SIZE = 0.4  # paper §4.1: 60% train / 40% test
N_TRIALS = 10  # paper §4.1: average of 10 independent trials (split seeds 0..9)
STANDARDIZE = False  # paper does not mention scaling; True = z-score with training stats
SUBSAMPLE = {"shuttle": 10_000}  # paper §4.1: Shuttle truncated to 10,000 random rows
SUBSAMPLE_SEED = 0

# the 14 ODDS (.mat) datasets of the paper's Table 1; the 16 DAMI (.arff) ones are not used
PAPER_DATASETS = [
    "arrhythmia", "breastw", "cardio", "ionosphere", "lympho", "mammography",
    "optdigits", "pima", "satellite", "satimage-2", "shuttle", "speech", "wbc", "wine",
]


def detectors(contamination, seed):
    """Fresh detector instances per trial. Params are PyOD defaults unless listed."""
    return {
        "ECOD": ECOD(contamination=contamination),  # PyOD implementation
        "ECOD-Alg1": ECODPaper(contamination=contamination),  # paper's Algorithm 1
        "HBOS": HBOS(contamination=contamination),
        "IForest": IForest(contamination=contamination, random_state=seed),
        "LOF": LOF(contamination=contamination),
        "KNN": KNN(contamination=contamination),
        "PCA": PCA(contamination=contamination, random_state=seed),
        "OCSVM": OCSVM(contamination=contamination),
        "LODA": LODA(contamination=contamination),
        "CBLOF": CBLOF(contamination=contamination, random_state=seed),
        "ABOD": ABOD(contamination=contamination),
    }


def load(name):
    """ODDS .mat from data/, or the ADBench copy of the ODDS file from data/adbench/."""
    mat = DATA_DIR / f"{name}.mat"
    if mat.exists():
        m = loadmat(mat)
    else:
        m = np.load(next((DATA_DIR / "adbench").glob(f"*_{name}.npz")), allow_pickle=False)
    X, y = m["X"].astype(float), m["y"].ravel().astype(int)
    if name in SUBSAMPLE:
        idx = np.random.default_rng(SUBSAMPLE_SEED).choice(len(X), SUBSAMPLE[name], replace=False)
        X, y = X[idx], y[idx]
    return X, y


def run_dataset(name, n_trials):
    X, y = load(name)
    contamination = y.mean()
    rows = []
    for trial in range(n_trials):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=trial, stratify=y
        )
        if STANDARDIZE:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for det_name, det in detectors(contamination, trial).items():
            t0 = time.perf_counter()
            try:
                det.fit(Xtr)
                scores = det.decision_function(Xte)
                auc = roc_auc_score(yte, scores)
                ap = average_precision_score(yte, scores)
            except Exception as e:  # keep going if one detector breaks on a dataset
                print(f"  {name}/{det_name} failed: {e}")
                auc = ap = np.nan
            rows.append(
                dict(dataset=name, trial=trial, detector=det_name, roc_auc=auc,
                     ap=ap, seconds=time.perf_counter() - t0)
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                    help="dataset names (default: the paper's 14 ODDS datasets)")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    names = args.datasets or PAPER_DATASETS
    all_rows = []
    for name in names:
        print(f"== {name}", flush=True)
        all_rows += run_dataset(name, args.trials)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "raw.csv", index=False)
    for metric in ("roc_auc", "ap", "seconds"):
        tab = df.pivot_table(index="dataset", columns="detector", values=metric, aggfunc="mean")
        tab.loc["MEAN"] = tab.mean()
        tab.round(4).to_csv(out / f"{metric}.csv")
        print(f"\n{metric} (mean over {args.trials} trials)")
        print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
