"""Test which ECOD variant (if any) reproduces the paper's ECOD numbers in Tables 4/5.

Variants: PyOD's ECOD, the paper's Algorithm 1, right-tail only, and both of the first
two with ECDFs estimated on the test split alone. Prints per-dataset ROC/AP next to the
paper and the mean (absolute) difference per variant.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.models.ecod import ECOD
from ecod_repro.ecod_paper import ECODPaper
import run_benchmark as rb

paper = pd.read_csv("data/paper_tables_4_5.csv").query("detector=='ECOD'").pivot(index="dataset", columns="metric", values="value")
def score(kind, Xtr, Xte):
    if kind.endswith("@test"):  # estimate ECDFs on the test split alone
        Xtr, kind = None, kind[:-5]
    mk = {"PyOD": lambda: ECOD(), "Alg1": lambda: ECODPaper(), "right": lambda: ECODPaper(variant="right"),
          "both": lambda: ECODPaper(variant="both"), "left": lambda: ECODPaper(variant="left")}[kind]
    d = mk()
    return d.fit(Xte).decision_scores_ if Xtr is None else d.fit(Xtr).decision_function(Xte)

kinds = ["PyOD", "Alg1", "right", "PyOD@test", "Alg1@test"]
out = []
for name in rb.PAPER_DATASETS:
    X, y = rb.load(name)
    for t in range(10):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=t, stratify=y)
        for k in kinds:
            s = score(k, Xtr, Xte)
            out.append((name, k, roc_auc_score(yte, s), average_precision_score(yte, s)))
df = pd.DataFrame(out, columns=["dataset", "kind", "roc_auc", "ap"]).groupby(["dataset", "kind"]).mean()
for m in ["roc_auc", "ap"]:
    tab = df[m].unstack()[kinds]
    tab.insert(0, "paper", paper[m])
    print(f"\n{m}\n", tab.round(3).to_string())
    err = tab[kinds].sub(tab["paper"], axis=0)
    print("mean diff ", err.mean().round(3).to_dict())
    print("mean |diff|", err.abs().mean().round(3).to_dict())
