"""Put our results next to the paper's Tables 4 (ROC) and 5 (AP) for the 14 ODDS datasets.

Reads results/raw.csv (from run_benchmark.py) and data/paper_tables_4_5.csv; writes
results/compare_<metric>.csv and prints the tables.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "results"


def main():
    ours = pd.read_csv(OUT / "raw.csv")
    paper = pd.read_csv(ROOT / "data" / "paper_tables_4_5.csv")

    for metric in ("roc_auc", "ap"):
        mine = ours.pivot_table(index="dataset", columns="detector", values=metric, aggfunc="mean")
        theirs = paper[paper.metric == metric].pivot(index="dataset", columns="detector", values="value")
        mine = mine.loc[theirs.index]

        # ECOD per dataset: paper vs PyOD ECOD vs Algorithm 1
        ecod = pd.DataFrame({
            "paper": theirs["ECOD"],
            "PyOD ECOD": mine["ECOD"],
            "Alg.1 ECOD": mine["ECOD-Alg1"],
        })
        ecod["diff (PyOD)"] = ecod["PyOD ECOD"] - ecod["paper"]
        ecod["diff (Alg.1)"] = ecod["Alg.1 ECOD"] - ecod["paper"]
        ecod.loc["MEAN"] = ecod.mean()

        # averages over the 14 datasets for every detector both of us ran
        common = [c for c in theirs.columns if c in mine.columns]
        avg = pd.DataFrame({"paper": theirs[common].mean(), "ours": mine[common].mean()})
        avg.loc["ECOD-Alg1"] = [float("nan"), mine["ECOD-Alg1"].mean()]
        avg["diff"] = avg["ours"] - avg["paper"]
        avg = avg.sort_values("ours", ascending=False)

        ecod.round(3).to_csv(OUT / f"compare_{metric}_ecod.csv")
        avg.round(3).to_csv(OUT / f"compare_{metric}_avg.csv")
        print(f"\n=== {metric}: ECOD per dataset ===\n{ecod.round(3).to_string()}")
        print(f"\n=== {metric}: mean over 14 ODDS datasets, all detectors ===\n{avg.round(3).to_string()}")


if __name__ == "__main__":
    main()
