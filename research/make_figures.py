"""
Builds the paper figures from research/results.

Run from the project folder:
    venv\\Scripts\\python.exe research\\make_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "research", "results")
FIG = os.path.join(ROOT, "research", "figures")
os.makedirs(FIG, exist_ok=True)

# Validated categorical palette (light mode), fixed order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)

    def box(x, y, w, h, text, fill="#eef4fc", edge=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=2, shrinkB=2))

    ax.text(1, 40, "Training pipeline (ZenML)", fontsize=9, fontweight="bold", va="top")
    steps = ["Data\ningestion", "Missing\nvalues", "Log\ntransform", "Outlier\nremoval",
             "Train/test\nsplit", "Model\ntraining", "Model\nevaluation"]
    w, h, gap, y = 11.4, 8, 2.6, 26
    xs = [1 + i * (w + gap) for i in range(len(steps))]
    for x, s in zip(xs, steps):
        box(x, y, w, h, s)
    for i in range(len(xs) - 1):
        arrow(xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(1, 17, "Deployment and inference", fontsize=9, fontweight="bold", va="top")
    y2 = 3
    box(xs[5] - 7, y2, w + 14, h, "MLflow tracking\n(params, metrics, model)", "#fdf0ea", ORANGE)
    box(xs[3] - 3, y2, w + 6, h, "MLflow model\nserver (REST)", "#fdf0ea", ORANGE)
    box(xs[0], y2, w + 10, h, "Inference pipeline\n(batch requests)", "#fdf0ea", ORANGE)
    arrow(xs[5] + w / 2, y, xs[5] + w / 2, y2 + h)
    arrow(xs[5] - 7, y2 + h / 2, xs[3] + w + 3, y2 + h / 2)
    arrow(xs[0] + w + 10, y2 + h / 2, xs[3] - 3, y2 + h / 2)
    save(fig, "fig1_architecture.png")


# ----------------------------------------------------------------------------
def fig_outliers():
    # keep_default_na=False: the rule literally named "None" must not become NaN
    e1 = pd.read_csv(os.path.join(RES, "e1_outliers.csv"), keep_default_na=False)
    e1["rule"] = e1.rule.replace({"None": "No removal"})
    rules = list(dict.fromkeys(e1.rule))
    protos = [("Filter before split", BLUE, "Filter before split (test set filtered)"),
              ("Filter training folds only", ORANGE, "Filter training folds only (all houses tested)")]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(rules))
    bw = 0.36
    floor = -0.3  # values below this are drawn as a stub and labelled "off scale"
    for j, (p, color, label) in enumerate(protos):
        d = e1[e1.protocol == p].set_index("rule").loc[rules]
        pos = x + (j - 0.5) * (bw + 0.03)
        vals, sds = d.R2_log_mean.values, d.R2_log_std.values
        shown = np.maximum(vals, floor)
        err = np.where(vals < floor, 0, sds)
        ax.bar(pos, shown, bw, yerr=err, color=color, label=label,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for xi, v, s in zip(pos, vals, sds):
            if v < floor:
                ax.text(xi + bw / 2 + 0.04, floor / 2, f"R² = {v:.2f}\n(off scale)", ha="left",
                        va="center", fontsize=7.3, color=INK)
            else:
                ax.text(xi, v + s + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.3, color=INK)
    labels = [f"{r}\n(n = {int(e1[e1.rule == r].rows_kept.iloc[0]):,})" for r in rules]
    ax.set_xticks(x, labels, fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(floor, 1.05)
    ax.set_ylabel("R² (log price), mean ± SD")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8)
    save(fig, "fig2_outlier_protocols.png")


# ----------------------------------------------------------------------------
def fig_models():
    e3 = pd.read_csv(os.path.join(RES, "e3_models.csv"))
    models = list(dict.fromkeys(e3.model))
    feats = [("Numeric only", BLUE), ("Numeric + categorical", ORANGE)]
    finite = e3[e3.RMSE_usd_mean < 100_000]
    cap = float(np.ceil(finite.RMSE_usd_mean.max() / 5000 + 1) * 5000)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    x = np.arange(len(models))
    bw = 0.36
    for j, (f, color) in enumerate(feats):
        d = e3[e3.features == f].set_index("model").loc[models]
        pos = x + (j - 0.5) * (bw + 0.03)
        vals = d.RMSE_usd_mean.values
        # Spread is reported in Table IV; error bars here hid the labels.
        shown = np.where(vals > cap, 0, vals)
        ax.bar(pos, shown, bw, color=color, label=f)
        for xi, v, s in zip(pos, vals, shown):
            if v <= cap:
                ax.text(xi, s + cap * 0.015, f"${v / 1000:,.1f}k", ha="center", va="bottom", fontsize=7.3, color=INK)
            else:
                ax.text(xi, cap * 0.03, f"not\ndrawn:\nunstable\nfit\n(RMSE ≈\n${v / 1e9:,.1f} bn)",
                        ha="center", va="bottom", fontsize=6.6, color=INK, linespacing=1.1)
    ax.set_xticks(x, models)
    ax.set_ylim(0, cap * 1.12)
    ax.set_ylabel("RMSE (US$), mean over 15 folds")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig3_model_comparison.png")


# ----------------------------------------------------------------------------
def fig_pred_vs_actual():
    info = json.load(open(os.path.join(RES, "summary.json")))
    best = f"{info['best_model']['model']} | {info['best_model']['features']}"
    oof = pd.read_csv(os.path.join(RES, "e3_oof_predictions.csv"))
    yt = np.expm1(oof.actual_log) / 1000
    yp = np.expm1(oof[best]) / 1000
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(yt, yp, s=9, color=BLUE, alpha=0.45, edgecolors="none")
    lim = [0, max(yt.max(), yp.max()) * 1.03]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.text(lim[1] * 0.97, lim[1] * 0.90, "perfect prediction", ha="right", fontsize=8, color=INK2, rotation=40)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual sale price (US$ thousands)")
    ax.set_ylabel("Predicted sale price (US$ thousands)")
    ax.set_aspect("equal")
    save(fig, "fig4_pred_vs_actual.png")


# ----------------------------------------------------------------------------
def fig_importance():
    imp = pd.read_csv(os.path.join(RES, "e4_importance.csv")).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(imp.feature, imp.importance_mean, xerr=imp.importance_std, color=BLUE, height=0.6,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    for y, (v, s) in enumerate(zip(imp.importance_mean, imp.importance_std)):
        ax.text(v + s + imp.importance_mean.max() * 0.03, y, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlabel("Increase in RMSE (log price) when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, imp.importance_mean.max() * 1.25)
    save(fig, "fig5_feature_importance.png")


if __name__ == "__main__":
    fig_architecture()
    fig_outliers()
    fig_models()
    fig_pred_vs_actual()
    fig_importance()
