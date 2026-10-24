# ECOD reproduction

Reproduction of Tables 4 and 5 of *ECOD: Unsupervised Outlier Detection Using Empirical
Cumulative Distribution Functions* (Li et al., IEEE TKDE 2022), using the authors' ECOD
implementation in PyOD on the 14 ODDS datasets of the paper.

```bash
uv sync                          # install dependencies
uv run python run_benchmark.py   # run the experiment, results in results/
uv run python compare.py         # compare with the paper's Tables 4 and 5
```

Data: ODDS (http://odds.cs.stonybrook.edu); breastw, mammography, speech and wine via
the ADBench copy of ODDS (`data/adbench/`).
