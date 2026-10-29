"""Corrected resampled t-test (Nadeau & Bengio, 2003) of the best model against every
other configuration, on several error metrics. Reads research/results/e3_models_per_fold.csv."""
import os
import numpy as np
import pandas as pd
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N, K = 2930, 5
pf = pd.read_csv(os.path.join(RES, "e3_models_per_fold.csv"))
best = ("XGBoost", "Numeric + categorical")
b = pf[(pf.model == best[0]) & (pf.features == best[1])].sort_values("fold")
rows = []
for (m, f), o in pf.groupby(["model", "features"], sort=False):
    if (m, f) == best:
        continue
    o = o.sort_values("fold")
    row = {"model": m, "features": f}
    for metric in ["RMSE_log", "MAE_usd", "RMSE_usd"]:
        d = o[metric].values - b[metric].values
        t = d.mean() / np.sqrt((1 / len(d) + (1 / K) / (1 - 1 / K)) * d.var(ddof=1))
        row[f"{metric}_diff"] = d.mean()
        row[f"{metric}_p"] = 2 * stats.t.sf(abs(t), df=len(d) - 1)
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "e3_significance_all.csv"), index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string())
