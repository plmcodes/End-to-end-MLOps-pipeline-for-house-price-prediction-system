"""
Experiments for the term paper.

Run from the project folder:
    venv\\Scripts\\python.exe research\\experiments.py

Outputs go to research/results (CSV/JSON) and research/figures (PNG).
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import stats
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "archive.zip")
RESULTS = os.path.join(ROOT, "research", "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 42
N_SPLITS, N_REPEATS = 5, 3
TARGET = "SalePrice"
IDS = ["Order", "PID"]
LOG_FEATURES = ["Gr Liv Area", TARGET]
Z = 3.0


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA)


def original_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly what the original pipeline does before outlier removal:
    mean-impute numeric columns on the FULL dataset, then log1p two columns."""
    df = df.copy()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].fillna(df[num].mean())
    for col in LOG_FEATURES:
        df[col] = np.log1p(df[col])
    return df


def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    # Same selection the original outlier step makes
    return df.select_dtypes(include=[int, float])


# ----------------------------------------------------------------------------
# Outlier rules. Each returns a boolean mask of rows to KEEP, using statistics
# computed only from `ref` (the data the rule is fitted on).
# ----------------------------------------------------------------------------
def keep_none(df, ref):
    return pd.Series(True, index=df.index)


def keep_global_z(df, ref):
    """Original rule: drop a row if ANY numeric column has |z| > 3."""
    r = numeric_only(ref)
    z = ((numeric_only(df)[r.columns] - r.mean()) / r.std()).abs()
    return ~(z > Z).any(axis=1)


def keep_target_z(df, ref):
    """Corrected rule: |z| > 3 on log(SalePrice) only."""
    mu, sd = ref[TARGET].mean(), ref[TARGET].std()
    return ((df[TARGET] - mu) / sd).abs() <= Z


def keep_target_iqr(df, ref):
    q1, q3 = ref[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[TARGET].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def keep_decock(df, ref):
    """De Cock (2011): remove houses with more than 4,000 sq ft of living area."""
    return df["Gr Liv Area"] <= np.log1p(4000)


OUTLIER_RULES = {
    "None": keep_none,
    "Global Z-score (original)": keep_global_z,
    "Target Z-score": keep_target_z,
    "Target IQR": keep_target_iqr,
    "De Cock (>4000 sq ft)": keep_decock,
}


# ----------------------------------------------------------------------------
# Metrics (y values are log1p(price))
# ----------------------------------------------------------------------------
PRED_LOG_CAP = 25.0  # expm1(25) is about $7e10; only reached by numerically unstable fits


def metrics(y_true_log, y_pred_log):
    # Unregularised OLS on one-hot features can produce astronomically large log
    # predictions; cap them so dollar metrics stay finite (count reported separately).
    n_capped = int((np.abs(y_pred_log) > PRED_LOG_CAP).sum())
    y_pred_log = np.clip(y_pred_log, -PRED_LOG_CAP, PRED_LOG_CAP)
    yt, yp = np.expm1(y_true_log), np.expm1(y_pred_log)
    ape = np.abs(yp - yt) / yt
    return {
        "R2_log": r2_score(y_true_log, y_pred_log),
        "RMSE_log": mean_squared_error(y_true_log, y_pred_log, squared=False),
        "R2_usd": r2_score(yt, yp),
        "RMSE_usd": mean_squared_error(yt, yp, squared=False),
        "MAE_usd": mean_absolute_error(yt, yp),
        "MAPE_pct": 100 * ape.mean(),
        "Pct_off_by_50pct": 100 * (ape > 0.5).mean(),
        "Max_APE_pct": 100 * ape.max(),
        "N_capped": n_capped,
    }


def summarize(rows, group_cols):
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in group_cols + ["repeat", "fold"]]
    agg = df.groupby(group_cols, sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return df, agg.reset_index()


def cv_splits(n):
    return list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(np.arange(n)))


# ----------------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------------
def original_model():
    """The original pipeline's model: mean imputer + LinearRegression."""
    return Pipeline([("imp", SimpleImputer(strategy="mean")), ("model", LinearRegression())])


def build_model(name, num_cols, cat_cols):
    linear = name in ("Linear Regression", "Ridge", "Lasso")
    num_steps = [("imp", SimpleImputer(strategy="mean"))]
    if linear:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        # In Ames, a missing category usually means "feature absent" (e.g. no pool)
        cat = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="None")),
            # Dense output: scikit-learn 1.3's RidgeCV sparse path is incompatible with SciPy >= 1.14
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat, cat_cols))
    pre = ColumnTransformer(transformers)

    if name == "Linear Regression":
        model = LinearRegression()
    elif name == "Ridge":
        model = RidgeCV(alphas=np.logspace(-2, 3, 30))
    elif name == "Lasso":
        model = LassoCV(n_alphas=40, cv=5, max_iter=20000, random_state=SEED, n_jobs=-1)
    elif name == "Random Forest":
        # max_features=1/3 is the usual choice for regression forests and is much faster
        model = RandomForestRegressor(n_estimators=200, max_features=0.33, n_jobs=-1, random_state=SEED)
    elif name == "XGBoost":
        model = XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.5, min_child_weight=2, reg_lambda=1.0,
            n_jobs=-1, random_state=SEED, tree_method="hist",
        )
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("model", model)])


# ----------------------------------------------------------------------------
# Experiment 0: reproduce the original reported number
# ----------------------------------------------------------------------------
def exp0_reproduce(raw):
    out = {}
    pre = original_preprocess(raw)
    for label, rule in [("original", keep_global_z), ("target_only_fix", keep_target_z)]:
        clean = numeric_only(pre)[rule(pre, pre)]
        X, y = clean.drop(columns=[TARGET]), clean[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
        m = original_model().fit(Xtr, ytr)
        p = m.predict(Xte)
        out[label] = {
            "rows_kept": int(len(clean)),
            "R2_log": r2_score(yte, p),
            "MSE_log": mean_squared_error(yte, p),
        }
    return out


# ----------------------------------------------------------------------------
# Experiment 1: outlier rule x where it is applied
# ----------------------------------------------------------------------------
def exp1_outliers(raw):
    pre = original_preprocess(raw)
    data = numeric_only(pre)
    rare_flag = ~keep_global_z(pre, pre)  # houses the original rule would delete
    rows, rare_rows, kept = [], [], {}

    for rule_name, rule in OUTLIER_RULES.items():
        kept[rule_name] = int(rule(pre, pre).sum())

        # (a) Original practice: filter the whole dataset, then split (test set is filtered too)
        clean = data[rule(pre, pre)]
        X, y = clean.drop(columns=[TARGET]), clean[TARGET]
        for i, (tr, te) in enumerate(cv_splits(len(clean))):
            m = original_model().fit(X.iloc[tr], y.iloc[tr])
            r = metrics(y.iloc[te].values, m.predict(X.iloc[te]))
            rows.append({"rule": rule_name, "protocol": "Filter before split", "fold": i, **r})

        # (b) Fair: filter only the training folds, test on every house in the fold
        X, y = data.drop(columns=[TARGET]), data[TARGET]
        for i, (tr, te) in enumerate(cv_splits(len(data))):
            tr_df = pre.iloc[tr]
            keep = rule(tr_df, tr_df).values
            m = original_model().fit(X.iloc[tr][keep], y.iloc[tr][keep])
            pred = m.predict(X.iloc[te])
            r = metrics(y.iloc[te].values, pred)
            rows.append({"rule": rule_name, "protocol": "Filter training folds only", "fold": i, **r})
            flag = rare_flag.iloc[te].values
            for grp, mask in [("Houses with a rare feature", flag), ("Other houses", ~flag)]:
                rr = metrics(y.iloc[te].values[mask], pred[mask])
                rare_rows.append({"rule": rule_name, "group": grp, "fold": i, **rr})

    per_fold, summary = summarize(rows, ["rule", "protocol"])
    summary["rows_kept"] = summary["rule"].map(kept)
    _, rare = summarize(rare_rows, ["rule", "group"])
    return per_fold, summary, rare, int(rare_flag.sum())


# ----------------------------------------------------------------------------
# Experiment 2: ablation ladder from the original pipeline to a corrected one
# ----------------------------------------------------------------------------
def exp2_ladder(raw):
    pre_full = original_preprocess(raw)
    raw_logged = raw.copy()
    for col in LOG_FEATURES:
        raw_logged[col] = np.log1p(raw_logged[col])  # stateless, safe before split
    cat_all = raw.select_dtypes(exclude="number").columns.tolist()
    rows = []
    splits = cv_splits(len(raw))

    def run(step, frame, rule, use_ids, use_cat, model_name, leak_impute):
        num = [c for c in frame.select_dtypes(include="number").columns if c != TARGET]
        if not use_ids:
            num = [c for c in num if c not in IDS]
        cats = cat_all if use_cat else []
        for i, (tr, te) in enumerate(splits):
            tr_df = frame.iloc[tr]
            keep = rule(tr_df, tr_df).values
            Xtr = tr_df[num + cats][keep]
            ytr = tr_df[TARGET][keep]
            if leak_impute:
                m = original_model().fit(Xtr, ytr)
            else:
                m = build_model(model_name, num, cats).fit(Xtr, ytr)
            r = metrics(frame.iloc[te][TARGET].values, m.predict(frame.iloc[te][num + cats]))
            rows.append({"step": step, "fold": i, **r})

    # L1: original choices, but evaluated fairly with repeated CV (filter train folds only)
    run("L1  Original pipeline, honest test set", pre_full, keep_global_z, True, False, None, True)
    run("L2  + Outlier rule on SalePrice only", pre_full, keep_target_z, True, False, None, True)
    run("L3  + Imputation fitted on training folds", raw_logged, keep_target_z, True, False, "Linear Regression", False)
    run("L4  + Drop ID columns (Order, PID)", raw_logged, keep_target_z, False, False, "Linear Regression", False)
    run("L5  + Categorical features (one-hot)", raw_logged, keep_target_z, False, True, "Linear Regression", False)
    run("L6  + Ridge regularisation", raw_logged, keep_target_z, False, True, "Ridge", False)
    return summarize(rows, ["step"])


# ----------------------------------------------------------------------------
# Experiment 3: model comparison under the corrected protocol
# ----------------------------------------------------------------------------
MODELS = ["Linear Regression", "Ridge", "Lasso", "Random Forest", "XGBoost"]


def corrected_frame(raw):
    df = raw.copy()
    for col in LOG_FEATURES:
        df[col] = np.log1p(df[col])
    num = [c for c in df.select_dtypes(include="number").columns if c not in IDS + [TARGET]]
    cat = df.select_dtypes(exclude="number").columns.tolist()
    return df, num, cat


def exp3_models(raw):
    df, num, cat = corrected_frame(raw)
    splits = cv_splits(len(df))
    rows, oof, timing = [], {}, {}
    for feat_name, cats in [("Numeric only", []), ("Numeric + categorical", cat)]:
        for name in MODELS:
            t0 = time.time()
            pred_rep0 = np.full(len(df), np.nan)
            for i, (tr, te) in enumerate(splits):
                tr_df = df.iloc[tr]
                keep = keep_target_z(tr_df, tr_df).values
                m = build_model(name, num, cats).fit(tr_df[num + cats][keep], tr_df[TARGET][keep])
                p = m.predict(df.iloc[te][num + cats])
                rows.append({"model": name, "features": feat_name, "fold": i, **metrics(df.iloc[te][TARGET].values, p)})
                if i < N_SPLITS:  # first repeat covers every house exactly once
                    pred_rep0[te] = p
            timing[(name, feat_name)] = (time.time() - t0) / len(splits)
            oof[f"{name} | {feat_name}"] = pred_rep0
            print(f"  {name:18s} {feat_name:22s} done in {time.time() - t0:5.1f}s", flush=True)
    per_fold, summary = summarize(rows, ["model", "features"])
    summary["fit_seconds_per_fold"] = [timing[(m, f)] for m, f in zip(summary["model"], summary["features"])]
    oof_df = pd.DataFrame(oof)
    oof_df["actual_log"] = df[TARGET].values
    return per_fold, summary, oof_df


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return t, p


def exp3_significance(per_fold, summary, n):
    best = summary.sort_values("RMSE_usd_mean").iloc[0]
    b = per_fold[(per_fold.model == best.model) & (per_fold.features == best.features)].sort_values("fold")
    out = []
    n_test = n / N_SPLITS
    for _, row in summary.iterrows():
        if row.model == best.model and row.features == best.features:
            continue
        o = per_fold[(per_fold.model == row.model) & (per_fold.features == row.features)].sort_values("fold")
        t, p = corrected_ttest(o.RMSE_usd.values, b.RMSE_usd.values, n - n_test, n_test)
        out.append({"model": row.model, "features": row.features,
                    "vs_best": f"{best.model} | {best.features}",
                    "RMSE_usd_diff": o.RMSE_usd.mean() - b.RMSE_usd.mean(), "t": t, "p_value": p})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Experiment 4: feature importance (permutation) for the best model
# ----------------------------------------------------------------------------
def exp4_importance(raw, model_name):
    df, num, cat = corrected_frame(raw)
    cols = num + cat
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    tr = tr[keep_target_z(tr, tr)]
    m = build_model(model_name, num, cat).fit(tr[cols], tr[TARGET])
    holdout = metrics(te[TARGET].values, m.predict(te[cols]))
    pi = permutation_importance(m, te[cols], te[TARGET], n_repeats=10, random_state=SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    return imp, holdout


# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    raw = load_raw()
    info = {"rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
            "numeric_columns": int(raw.select_dtypes(include="number").shape[1]),
            "categorical_columns": int(raw.select_dtypes(exclude="number").shape[1]),
            "missing_cells": int(raw.isna().sum().sum()),
            "columns_with_missing": int((raw.isna().sum() > 0).sum()),
            "price_min": float(raw[TARGET].min()), "price_max": float(raw[TARGET].max()),
            "price_median": float(raw[TARGET].median()), "price_mean": float(raw[TARGET].mean()),
            "price_skew": float(raw[TARGET].skew()),
            "log_price_skew": float(np.log1p(raw[TARGET]).skew()),
            "cv": f"{N_SPLITS}-fold x {N_REPEATS} repeats", "seed": SEED}

    print("E0: reproducing original result", flush=True)
    info["exp0"] = exp0_reproduce(raw)

    print("E1: outlier strategies", flush=True)
    e1_folds, e1, e1_rare, n_rare = exp1_outliers(raw)
    info["houses_flagged_by_global_z"] = n_rare
    e1.to_csv(os.path.join(RESULTS, "e1_outliers.csv"), index=False)
    e1_rare.to_csv(os.path.join(RESULTS, "e1_rare_houses.csv"), index=False)

    print("E2: ablation ladder", flush=True)
    _, e2 = exp2_ladder(raw)
    e2.to_csv(os.path.join(RESULTS, "e2_ladder.csv"), index=False)

    print("E3: model comparison", flush=True)
    e3_folds, e3, oof = exp3_models(raw)
    e3.to_csv(os.path.join(RESULTS, "e3_models.csv"), index=False)
    e3_folds.to_csv(os.path.join(RESULTS, "e3_models_per_fold.csv"), index=False)
    oof.to_csv(os.path.join(RESULTS, "e3_oof_predictions.csv"), index=False)
    sig = exp3_significance(e3_folds, e3, len(raw))
    sig.to_csv(os.path.join(RESULTS, "e3_significance.csv"), index=False)

    best = e3.sort_values("RMSE_usd_mean").iloc[0]
    info["best_model"] = {"model": best.model, "features": best.features}
    print(f"E4: permutation importance for {best.model} ({best.features})", flush=True)
    imp, holdout = exp4_importance(raw, best.model)
    imp.to_csv(os.path.join(RESULTS, "e4_importance.csv"), index=False)
    info["exp4_holdout"] = holdout

    info["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(info, f, indent=2, default=float)
    print(f"Done in {info['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
