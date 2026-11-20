"""Publication figures (revised composite system, 2026-11-08); rendering only, no model fitting.

Main figures
  Fig. 1  Nested literature data and the transfer gap
  Fig. 2  Condition-dependent study-residual calibration (schematic, residual structure, ablation)
  Fig. 3  Few-shot accuracy against constant offset and mixed-effect boosting
  Fig. 4  Interval calibration and label-free anchor design
  Fig. 5  Within-study process contrasts and cross-phase residual coupling
Supplementary
  Fig. S1 21-regressor benchmark; Fig. S2 base-learner sensitivity
Every figure folder holds the combined PNG/PDF, a notebook entry point and independently redrawn panels
in subfigures/ (not crops of the combined figure).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from publication_figure_utils import *
from common import RES
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Ellipse, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

C_BLUE, C_RED, C_GREEN, C_PALE_GREEN, C_LIGHT_BLUE, C_PINK, C_LIGHT_PINK, C_PALE_BLUE = PAL
_base_style_axis = style_axis


def style_axis(ax, xlabel=None, ylabel=None, tick_size=10.5, label_size=12):
    """Compact defaults for the three-column composite rows (2026-11-13)."""
    _base_style_axis(ax, xlabel, ylabel, tick_size, label_size)

TLAB = {"Y_oil": "Biocrude yield", "Y_char": "Solid/char yield", "Y_aq": "Aqueous yield", "Y_gas": "Gas yield",
        "ER_oil": "Calculated\nenergy ratio", "HHV_oil": "Biocrude HHV"}
SHORT = {"Y_oil": "Biocrude", "Y_char": "Char", "Y_aq": "Aqueous", "Y_gas": "Gas", "ER_oil": "E ratio", "HHV_oil": "HHV$_{oil}$"}
ORDER = ["Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "HHV_oil"]
R2 = "R$^{2}$"; Q2 = "Q$^{2}$"
DIV = LinearSegmentedColormap.from_list("blue_white_rose", [C_BLUE, "#FFFFFF", C_RED])
LETTER = dict(fontsize=15, fontweight="normal", color=INK)

# ------------------------------------------------------------------ data
q = pd.read_csv(RES / "02_quick_baseline.csv"); q = q[q.features == "chem+process"]
conf = pd.read_csv(RES / "06_study_conformal.csv")
sr = pd.read_csv(RES / "09_srgp_summary_oil_char.csv")
sec = pd.read_csv(RES / "09_srgp_summary_Y_aq_Y_gas_ER_oil_HHV_oil.csv")
vg = {t: pd.read_csv(RES / f"09_variogram_{t}.csv") for t in ["Y_oil", "Y_char"]}
theta = {t: pd.read_csv(RES / f"09_srgp_theta_{t}.csv") for t in ORDER}
raw = {t: pd.read_csv(RES / f"09_srgp_raw_{t}.csv") for t in ["Y_oil", "Y_char"]}
dml = pd.read_csv(RES / "04_fe_dml_effects_v2.csv")
lab = json.loads((RES / "10_lab_effect.json").read_text(encoding="utf-8"))
act = pd.read_csv(RES / "11_active_summary_Y_oil_Y_char.csv")
gpb = pd.read_csv(RES / "14_gpboost_summary_six_targets.csv")
gap = pd.read_csv(RES / "18_gpboost_gap_by_target_tuned.csv")
gapj = json.loads((RES / "18_gpboost_gap_vs_transfer_tuned.json").read_text())
clo = json.loads((RES / "15_closure_null.json").read_text())
nn = json.loads((RES / "19_nn_leakage.json").read_text())
nnd = pd.read_csv(RES / "19_nn_distances.csv")
tsne = pd.read_csv(RES / "19_tsne_records.csv")
bench = pd.read_csv(RES / "05_benchmark20.csv")
SUB_COL = {"HTL": C_BLUE, "HTL_solvothermal": C_PALE_GREEN, "HTC": C_RED}
SUB_LAB = {"HTL": "Hydrothermal liquefaction", "HTL_solvothermal": "Solvothermal liquefaction", "HTC": "Hydrothermal carbonization"}


# ================================================================== Fig. 1
def f1a(ax):
    """Records in predictor space (t-SNE, display only); rings mark the six largest source studies."""
    for s_, col in SUB_COL.items():
        z = tsne[tsne.process_subtype == s_]
        ax.scatter(z.x, z.y, s=7, color=col, alpha=0.55, lw=0, rasterized=True, label=SUB_LAB[s_])
    big = tsne.groupby("study_id").size().sort_values(ascending=False).index[:6]
    for s_ in big:
        z = tsne[tsne.study_id == s_]
        ax.scatter(z.x, z.y, s=20, facecolor="none", edgecolor=INK, lw=0.55, rasterized=True)
    ax.scatter([], [], s=20, facecolor="none", edgecolor=INK, lw=0.8, label="Six largest source studies")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    a = nn["all"]
    ax.text(0.99, 0.02, f"{a['records']:,} records · {a['studies']} source studies\n"
            f"nearest record strictly closer within study: {a['share_same_study_strictly_closer_all_records']:.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=INK,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=9.5, markerscale=1.6, handletextpad=0.2,
              borderaxespad=0.15, columnspacing=0.7, ncol=2)


def f1b(ax, legend=False):
    bins = [0, 1995, 2005, 2010, 2015, 2020, 2026]; labs = ["≤1995", "1996–05", "2006–10", "2011–15", "2016–20", "2021–26"]
    t = tsne.copy(); t["bin"] = pd.cut(t.year, bins, labels=labs)
    ct = t.groupby(["bin", "process_subtype"], observed=False).size().unstack(fill_value=0)[list(SUB_COL)]
    ns = t.groupby("bin", observed=False).study_id.nunique()
    x = np.arange(len(labs)); bottom = np.zeros(len(labs))
    for s_, col in SUB_COL.items():
        ax.bar(x, ct[s_].values, 0.72, bottom=bottom, color=col, edgecolor=INK, lw=0.7, label=SUB_LAB[s_])
        bottom += ct[s_].values
    for xi, b, n in zip(x, bottom, ns.values):
        ax.text(xi, b + 25, f"{n}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x)
    style_axis(ax, xlabel="Publication year", ylabel="Records", tick_size=10.5)
    ax.set_xticklabels(labs, fontsize=9.6, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, bottom.max() * 1.15); numeric_ticks(ax, x=False, n=4)
    if legend:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.12, 1.01), fontsize=9.0, handlelength=1.3, labelspacing=0.25)


def f1c(ax):
    for col_, col, ls, lb in [("nn_same_study", C_BLUE, "-", "Nearest record, same study"), ("nn_other_study", C_RED, "--", "Nearest record, other studies")]:
        v = np.sort(nnd[col_].values); e = np.arange(1, len(v) + 1) / len(v)
        ax.step(np.r_[0, v], np.r_[0, e], where="post", color=col, lw=2.4, ls=ls, label=lb)
    z0 = (nnd.nn_same_study < 1e-9).mean()
    ax.annotate(f"Identical predictors\nwithin study: {z0:.0%}", xy=(0.02, z0), xytext=(0.55, z0 + 0.19), fontsize=9.3, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    a = nn["all"]
    ax.text(0.98, 0.04, f"Random-row split: {a['random_split_test_nearest_train_same_study']:.0%}\nsame-study nearest\ntraining record",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.2, color=INK)
    style_axis(ax, xlabel="Distance in predictor space", ylabel="Cumulative share")
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.02); numeric_ticks(ax, n=4)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 1.0), fontsize=9.4, handlelength=1.8, labelspacing=0.2)


def f1d1(ax):
    seg = [("between_study", C_RED, "Between studies"), ("within_transferable(new study)", C_GREEN, "Within, transferable"),
           ("within_study_specific", C_BLUE, "Within, random-row gain"), ("within_unexplained", C_PINK, "Within, unexplained")]
    ys = [1, 0]; names = ["Biocrude", "Char"]
    for y, t in zip(ys, ["Y_oil", "Y_char"]):
        left = 0
        for key, col, lb in seg:
            v = lab[t]["budget"][key]
            ax.barh(y, v, 0.55, left=left, color=col, edgecolor=INK, lw=0.8, label=lb if t == "Y_oil" else None)
            if v > 0.12:
                ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=10.5, color=INK)
            left += v
    ax.set_yticks(ys); ax.set_yticklabels(names)
    style_axis(ax, xlabel="Share of yield variance")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.5); numeric_ticks(ax, y=False, n=5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 1.0), ncol=2, fontsize=9.3, handlelength=1.0, columnspacing=0.6,
              handletextpad=0.3, borderaxespad=0.1)


def f1f(ax):
    x = np.arange(2); w = 0.36
    for j, (key, col, lb) in enumerate([("ICC_raw", C_BLUE, "Observed"), ("ICC_resid", C_RED, "Held-out residual")]):
        vals = [lab[t][key] for t in ("Y_oil", "Y_char")]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, edgecolor=INK, lw=0.9, label=lb)
    ax.set_xticks(x); ax.set_xticklabels(["Biocrude", "Char"])
    style_axis(ax, ylabel="ICC")
    ax.set_ylim(0, 1.0); numeric_ticks(ax, x=False, n=5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, fontsize=9.2, handlelength=1.0,
              handletextpad=0.35, columnspacing=0.85, borderaxespad=0.1)


def f1e(ax):
    for i, t in enumerate(ORDER[::-1]):
        r = q[(q.target == t) & (q.split == "random")].R2.values[0]
        g = q[(q.target == t) & (q.split == "group")].R2.values[0]
        ax.plot([g, r], [i, i], color=INK, lw=1.3, zorder=1)
        ax.scatter(r, i, s=80, color=ROLE["random"], edgecolor=INK, lw=0.9, zorder=3, label="Random rows" if i == 0 else None)
        ax.scatter(g, i, s=80, color=ROLE["group"], edgecolor=INK, lw=0.9, marker="s", zorder=3, label="Held-out studies" if i == 0 else None)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([TLAB[t] for t in ORDER[::-1]], fontsize=10.5)
    ax.axvline(0, color=INK, lw=0.9, ls=":")
    style_axis(ax, xlabel=f"Cross-validated {R2}", tick_size=10.5)
    ax.set_yticklabels([TLAB[t] for t in ORDER[::-1]], fontsize=10.5)
    ax.set_xlim(-0.2, 1.0); numeric_ticks(ax, y=False, n=5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.3, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8, fontsize=10, borderaxespad=0.1)


# ================================================================== Fig. 2
def _box(ax, x, y, w, h, fc, text=None, fs=11, alpha=0.45):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=INK, lw=1.0, alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="none", ec=INK, lw=1.0))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, lw=1.4, color=INK))


def f2a(ax):
    """Editable explanatory workflow; all icons are vector primitives."""
    ax.set_xlim(0, 17.2); ax.set_ylim(0, 4.4); ax.axis("off")
    # Literature -> predictor/residuals. Pages represent source records, not a
    # claim that one apparatus was used throughout the literature compilation.
    ax.text(1.65, 4.05, "Historical studies", ha="center", fontsize=11.5, color=INK)
    for i, fill in enumerate([C_PALE_BLUE, C_LIGHT_BLUE, "white"]):
        x, y = 0.50 + i*.20, 2.15 - i*.12
        ax.add_patch(Polygon([(x,y),(x+1.25,y),(x+1.25,y+1.30),(x+.98,y+1.58),(x,y+1.58)],
                             fc=fill, ec=INK, lw=.8))
        for yy in [.40,.70,1.00]:
            ax.plot([x+.20,x+1.02],[y+yy]*2,color=C_BLUE,lw=1.2)
    ax.add_patch(Ellipse((2.38,2.55),.85,.40,angle=38,fc=C_PALE_GREEN,ec=INK,lw=.7))
    ax.plot([2.04,2.70],[2.28,2.79],color=INK,lw=.7)
    ax.text(1.65, 1.25, "Pooled learner $f(x)$", ha="center", fontsize=10.1, color=INK)
    ax.text(1.65, .64, "$r = y-f(x)$", ha="center", fontsize=11.1, color=INK)
    _arrow(ax, 3.10, 2.45, 3.80, 2.45)
    # Covariance is estimated from held-out residuals from the source studies.
    ax.text(6.55, 4.05, "Shared residual covariance", ha="center", fontsize=11.5, color=INK)
    for w, h, c in [(2.05,1.0,C_PALE_BLUE),(1.45,.70,C_LIGHT_BLUE),(.80,.38,C_PINK)]:
        ax.add_patch(Ellipse((6.55,2.95),w,h,angle=12,fc=c,ec=INK,lw=.7))
    ax.text(6.55, 1.88, "$k(x,x')=\\sigma_u^2+\\sigma_v^2e^{-d^2/(2\\ell^2)}\\rho^m+\\sigma_e^2\\delta$",
            ha="center", va="center", fontsize=11.0, color=INK)
    ax.text(6.55, 1.12, "$d$: condition distance; $m$: category mismatches", ha="center", fontsize=9.0, color=INK)
    ax.text(6.55, .55, "Fit to held-out residuals", ha="center", fontsize=9.5, color=INK)
    _arrow(ax, 9.22, 2.45, 9.88, 2.45)
    ax.text(13.62, 4.05, "New study: $k$ anchors", ha="center", fontsize=11.5, color=INK)
    ins = ax.inset_axes([10.43, .76, 3.78, 2.76], transform=ax.transData)
    xs = np.linspace(0, 1, 200)
    f = 30 + 14 * xs - 10 * xs ** 2
    true = f + 2 + 9 * np.exp(-((xs - 0.25) / 0.18) ** 2) - 3 * xs
    A = np.array([0.18, 0.32, 0.86]); ya = np.interp(A, xs, true)
    ra = ya - np.interp(A, xs, f)
    k = lambda a, b: 2.0 + 30 * np.exp(-(a[:, None] - b[None]) ** 2 / (2 * 0.12 ** 2))
    Kaa = k(A, A) + np.eye(3); Kxa = k(xs, A)
    mu = Kxa @ np.linalg.solve(Kaa, ra)
    var = 32 - np.einsum("ij,ji->i", Kxa, np.linalg.solve(Kaa, Kxa.T)) + 1
    sd = np.sqrt(np.maximum(var, .1))
    ins.fill_between(xs, f+mu-1.64*sd, f+mu+1.64*sd, color=C_PINK, alpha=.55, lw=0)
    ins.plot(xs,true,color=INK,lw=1,ls=":",label="Study response")
    ins.plot(xs,f,color=C_BLUE,lw=2,ls="--",label="Pooled $f$")
    ins.plot(xs,f+ra.mean(),color=C_PALE_GREEN,lw=2,ls="-.",label="Constant offset")
    ins.plot(xs,f+mu,color=C_RED,lw=2,label="Local correction")
    ins.scatter(A,ya,marker="*",s=95,fc="white",ec=INK,lw=.8,zorder=5,label="Anchors")
    ins.set_xticks([]); ins.set_yticks([])
    ins.spines[["top","right"]].set_visible(False)
    ins.set_xlabel("Operating condition",fontsize=9.5,labelpad=2)
    ins.set_ylabel("Yield",fontsize=10,labelpad=2)
    ins.legend(loc="center left",bbox_to_anchor=(1.02,.50),fontsize=8.6,
               handlelength=1.35,borderaxespad=0,labelspacing=.45,frameon=False)


def _variogram(ax, t, legend=False):
    v = vg[t]; th = theta[t][["full_su", "full_sv", "full_l", "full_rho", "full_se"]].median()
    su, sv, l, rho, se = th.values
    for mm, col, mk, lb in [(False, C_BLUE, "o", "Empirical, m = 0"), (True, C_RED, "^", "Empirical, m ≥ 1")]:
        z = v[v.any_mismatch == mm].sort_values("dist")
        ax.scatter(z.dist, z.gamma, s=46, marker=mk, color=col, edgecolor=INK, lw=0.8, zorder=3, label=lb)
    d = np.linspace(0, v.dist.max() * 1.05, 200)
    for m, ls, lb in [(0, "-", "Fitted, m = 0"), (1, "--", "Fitted, m = 1"), (2, "-.", "Fitted, m = 2")]:
        gam = sv ** 2 * (1 - np.exp(-d ** 2 / (2 * l ** 2)) * rho ** m) + se ** 2
        ax.plot(d, gam, color=C_BLUE, lw=1.8, ls=ls, label=lb)
    ax.axhline(v.sill_between.iloc[0], color=INK, lw=0.9, ls=(0, (1, 3)), label="Between studies")
    phase = "Biocrude" if t == "Y_oil" else "Char"
    style_axis(ax, xlabel="Condition distance $d$", ylabel=f"{phase} semivariance (pp$^2$)", label_size=12.0)
    ax.set_ylim(0, max(v.sill_between.iloc[0], v.gamma.max(), gam.max()) * 1.06); numeric_ticks(ax, n=4)
    if legend:
        handles,labels=ax.get_legend_handles_labels()
        pick=range(2) if legend=="empirical" else (range(2,6) if legend=="fitted" else range(6))
        ax.legend([handles[i] for i in pick],[labels[i] for i in pick],loc="lower center",
                  bbox_to_anchor=(0.5,1.01),fontsize=9.0,ncol=1 if legend=="empirical" else 2,
                  handlelength=1.5,labelspacing=.25,columnspacing=.7,handletextpad=.35,borderaxespad=.2)


def f2b(ax): _variogram(ax, "Y_oil", legend="empirical")
def f2c(ax): _variogram(ax, "Y_char", legend="fitted")


def f2d(ax):
    rows = []
    for t in ORDER:
        th = theta[t][["full_su", "full_sv", "full_se"]].median()
        v = np.array([th.full_su ** 2, th.full_sv ** 2, th.full_se ** 2]); rows.append(v / v.sum())
    rows = np.array(rows)
    y = np.arange(len(ORDER))[::-1]; left = np.zeros(len(ORDER))
    for j, (col, lb) in enumerate([(C_LIGHT_BLUE, "Study intercept $\\sigma_u^2$"), (C_RED, "Condition-dependent $\\sigma_v^2$"), (C_GREEN, "Run-level $\\sigma_e^2$")]):
        ax.barh(y, rows[:, j], 0.62, left=left, color=col, edgecolor=INK, lw=0.8, label=lb)
        left += rows[:, j]
    ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in ORDER])
    style_axis(ax, xlabel="Variance share")
    ax.set_yticklabels([SHORT[t] for t in ORDER])
    ax.set_xlim(0, 1); numeric_ticks(ax, y=False, n=5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.25, 1.0), ncol=1, fontsize=9.5, handlelength=1.0, labelspacing=0.25)


def f2e(ax):
    meths = [("offset", "Constant offset"), ("BLUP(fixed)", "Fixed shrinkage"), ("kernel(fixed)", "Fixed distance kernel"),
             ("SR-GP(no local)", "Intercept only"), ("SR-GP(no intercept)", "Local covariance only"), ("SR-GP", "Full SR-GP")]
    y = np.arange(len(meths))[::-1]
    for t, col, mk, dy in [("Y_oil", C_BLUE, "o", 0.14), ("Y_char", C_RED, "s", -0.14)]:
        z = sr[(sr.target == t) & (sr.k == 5)].set_index("model").R2
        vals = [z[m] for m, _ in meths]
        ax.scatter(vals, y + dy, s=70, marker=mk, color=col, edgecolor=INK, lw=0.9, zorder=3, label="Biocrude" if t == "Y_oil" else "Char")
    ax.set_yticks(y); ax.set_yticklabels([lb for _, lb in meths])
    style_axis(ax, xlabel=f"{Q2} with five anchor runs")
    ax.set_yticklabels([lb for _, lb in meths])
    ax.set_xlim(0.55, 0.85); numeric_ticks(ax, y=False, n=4)
    ax.legend(loc="lower left", fontsize=10)


# ================================================================== Fig. 3
def _matched(ax, t, legend):
    methods = [("offset", C_BLUE, "o", "--", "Constant offset"), ("GPB-RE", C_GREEN, "s", ":", "GPBoost + intercept"),
               ("GPB-RE+GP", C_PINK, "D", "-.", "GPBoost + local GP"), ("SR-GP", C_RED, "^", "-", "SR-GP")]
    for m, col, mk, ls, lb in methods:
        z = gpb[(gpb.target == t) & (gpb.model == m)].sort_values("k")
        ax.plot(z.k, z.R2, color=col, marker=mk, ls=ls, lw=2, ms=7, mec=INK, mew=0.8, label=lb)
    ax.set_xticks([1, 3, 5]); ax.set_ylim(0.35, 0.85)
    phase = "Biocrude" if t == "Y_oil" else "Char"
    style_axis(ax, xlabel="Anchor runs", ylabel=f"{phase} query {Q2}")
    numeric_ticks(ax, x=False, n=5)
    if legend:
        ax.legend(loc="lower right", fontsize=8.9, handlelength=1.4, labelspacing=0.25, borderaxespad=0.1)


def f3a(ax): _matched(ax, "Y_oil", True)
def f3b(ax): _matched(ax, "Y_char", False)


def f3c(ax):
    y = np.arange(len(ORDER))[::-1]
    for (lbl, col, mk) in [("Constant offset", C_BLUE, "o"), ("Better GPBoost", C_PINK, "D"), ("SR-GP", C_RED, "^")]:
        vals = []
        for t in ORDER:
            z = gpb[(gpb.target == t) & (gpb.k == 5)].set_index("model").R2
            vals.append(z["offset"] if lbl == "Constant offset" else (max(z["GPB-RE"], z["GPB-RE+GP"]) if lbl == "Better GPBoost" else z["SR-GP"]))
        ax.scatter(vals, y, s=70, marker=mk, color=col, edgecolor=INK, lw=0.9, zorder=3, label=lbl)
    for yi, t in zip(y, ORDER):
        z = gpb[(gpb.target == t) & (gpb.k == 5)].set_index("model").R2
        v = [z["offset"], max(z["GPB-RE"], z["GPB-RE+GP"]), z["SR-GP"]]
        ax.plot([min(v), max(v)], [yi, yi], color=INK, lw=1.0, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in ORDER])
    style_axis(ax, xlabel=f"{Q2} with five anchor runs")
    ax.set_yticklabels([SHORT[t] for t in ORDER])
    ax.set_xlim(0.58, 0.92); numeric_ticks(ax, y=False, n=4)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, fontsize=9.5, handletextpad=0.2, columnspacing=0.6)


def f3d(ax):
    z1 = gap[gap.k == 1].set_index("target").reindex(ORDER); z5 = gap[gap.k == 5].set_index("target").reindex(ORDER)
    r1 = gapj["by_k"]["1"]["gap_best~Q2_pooled_k0"]["pearson"]; r5 = gapj["by_k"]["5"]["gap_best~Q2_pooled_k0"]["pearson"]
    ax.errorbar(z1.Q2_pooled_k0, z1.gap_best, yerr=[z1.gap_best - z1.gap_best_lo, z1.gap_best_hi - z1.gap_best], fmt="o", ms=8,
                color=C_RED, mec=INK, mew=0.9, ecolor=INK, elinewidth=0.9, capsize=2.5, label=f"One anchor, r = {r1:.2f}", zorder=3)
    ax.scatter(z5.Q2_pooled_k0, z5.gap_best, s=55, marker="s", facecolor="white", edgecolor=C_BLUE, lw=1.6, zorder=2, label=f"Five anchors, r = {r5:.2f}")
    xx = np.linspace(-0.06, 0.48, 50); b = np.polyfit(z1.Q2_pooled_k0, z1.gap_best, 1)
    ax.plot(xx, np.polyval(b, xx), color=C_RED, lw=1.2, ls="--", zorder=1)
    off = {"Y_oil": (0.04, -0.045), "Y_char": (0.015, 0.012), "Y_aq": (0.012, -0.03), "Y_gas": (0.078, -0.046), "ER_oil": (-0.11, 0.03), "HHV_oil": (0.014, -0.032)}
    for t, xv, yv in zip(ORDER, z1.Q2_pooled_k0, z1.gap_best):
        ax.text(xv + off[t][0], yv + off[t][1], SHORT[t], fontsize=9.2, color=INK)
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    style_axis(ax, xlabel=f"Zero-anchor pooled {Q2}", ylabel=f"SR-GP − better GPBoost ({Q2})")
    ax.set_xlim(-0.08, 0.55); numeric_ticks(ax, n=4)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), fontsize=9.5, handletextpad=0.3, borderaxespad=0.1)


def f3e(ax):
    z = gap[gap.k == 5].set_index("target").reindex(ORDER)
    y = np.arange(len(ORDER))[::-1]
    for col_, lo, hi, mk, col, dy, lb in [("gap_best", "gap_best_lo", "gap_best_hi", "o", C_RED, 0.15, "SR-GP vs better GPBoost"),
                                          ("gap_kernel", "gap_kernel_lo", "gap_kernel_hi", "s", C_PALE_GREEN, -0.15, "SR-GP@GPBoost vs GPBoost + local GP")]:
        ax.errorbar(z[col_], y + dy, xerr=[z[col_] - z[lo], z[hi] - z[col_]], fmt=mk, ms=7, color=col, mec=INK, mew=0.9,
                    ecolor=INK, elinewidth=1.0, capsize=2.5, label=lb)
    ax.axvline(0, color=INK, lw=0.9, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in ORDER])
    style_axis(ax, xlabel=f"Difference in {Q2}, five anchors")
    ax.set_yticklabels([SHORT[t] for t in ORDER])
    numeric_ticks(ax, y=False, n=4)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.12, 1.0), ncol=1, fontsize=8.4, handletextpad=0.25, borderaxespad=0.1)


def f3f(ax, t="Y_oil", k=3):
    r = raw[t]; r = r[r.k == k]
    st = r.groupby(["study", "model"]).rmse_study.mean().unstack()
    ax.scatter(st["offset"], st["SR-GP"], s=34, facecolor="none", edgecolor=C_RED, lw=1.1)
    lim = [0, np.nanmax(st[["offset", "SR-GP"]].values) * 1.05]
    ax.plot(lim, lim, color=INK, lw=1.0, ls="--")
    win = np.mean(st["SR-GP"] < st["offset"])
    ax.text(0.04, 0.96, f"SR-GP lower in {win:.0%}\nof biocrude studies", transform=ax.transAxes, va="top", fontsize=10.5,
            bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_xlim(lim); ax.set_ylim(lim)
    style_axis(ax, xlabel="Study RMSE, offset (pp)", ylabel="Study RMSE, SR-GP (pp)")
    numeric_ticks(ax, n=4)


# ================================================================== Fig. 4
def _design(ax, t, legend):
    for stg, col, mk, ls, lb in [("random", C_BLUE, "o", "--", "Random anchors"), ("max-min distance", C_GREEN, "D", ":", "Space-filling anchors"),
                                 ("IVR (active)", C_RED, "^", "-", "Variance-selected anchors")]:
        z = act[(act.target == t) & (act.strategy == stg)].sort_values("k")
        ax.plot(z.k, z.R2, color=col, ls=ls, lw=2, marker=mk, ms=7.5, mec=INK, mew=0.8, label=lb)
    r5 = act[(act.target == t) & (act.strategy == "random") & (act.k == 5)].R2.iloc[0]
    ax.axhline(r5, color=INK, lw=0.9, ls=(0, (2, 2)))
    phase = "Biocrude" if t == "Y_oil" else "Char"
    style_axis(ax, xlabel="Anchor runs", ylabel=f"{phase} query {Q2}")
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_ylim(0.48, 0.86); numeric_ticks(ax, x=False, n=5)
    if legend:
        ax.legend(loc="lower right", fontsize=9.6, handlelength=1.8, labelspacing=0.3)


def f4a(ax): _design(ax, "Y_oil", True)
def f4b(ax): _design(ax, "Y_char", False)


def f4c(ax, legend_columns=2):
    for t, mk in [("Y_oil", "o"), ("Y_char", "s")]:
        for m, col, lb in [("offset", C_BLUE, "Constant offset"), ("SR-GP", C_RED, "SR-GP")]:
            z = sr[(sr.target == t) & (sr.model == m)].sort_values("k")
            ax.plot(z.width, z.coverage, color=col, marker=mk, ms=8, lw=1.6, mec=INK, mew=0.8,
                    label=f"{lb}, {'biocrude' if t == 'Y_oil' else 'char'}")
    ax.axhline(0.9, color=INK, ls="--", lw=1.1)
    style_axis(ax, xlabel="Mean interval width (pp)", ylabel="Empirical coverage")
    ax.set_ylim(0.86, 0.93); ax.set_xlim(24, 46); numeric_ticks(ax, n=5)
    ax.legend(loc="lower center" if legend_columns == 2 else "lower right", fontsize=9.4, ncol=legend_columns,
              labelspacing=0.3, columnspacing=1.05)


def f4d(ax):
    pos, data, cols = [], [], []
    for i, t in enumerate(["Y_oil", "Y_char"]):
        r = raw[t]; r = r[(r.k == 5) & r.cover.notna()]
        for j, (m, col) in enumerate([("offset", C_BLUE), ("SR-GP", C_RED)]):
            z = r[r.model == m]
            s = z.groupby("study").apply(lambda g: (g.cover * g.n).sum() / g.n.sum(), include_groups=False)
            pos.append(i * 3 + j); data.append(s.values); cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.62, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color=INK), capprops=dict(color=INK))
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c); b.set_edgecolor(INK)
    rng = np.random.default_rng(5)
    for p, d_, c in zip(pos, data, cols):
        ax.scatter(p + rng.uniform(-0.18, 0.18, len(d_)), d_, s=9, color=c, edgecolor=INK, lw=0.3, alpha=0.8, zorder=3)
    for i, t in enumerate(["Y_oil", "Y_char"]):
        v = conf[(conf.target == t) & (conf.method == "naive_row")].study_cov_median.iloc[0]
        ax.scatter(i * 3 + 0.5, v, s=90, marker="X", color=C_LIGHT_PINK, edgecolor=INK, lw=0.9, zorder=4,
                   label="Row-calibrated, study median" if i == 0 else None)
    ax.axhline(0.9, color=INK, ls="--", lw=1.0)
    ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(["Biocrude", "Char"])
    style_axis(ax, ylabel="Study coverage")
    ax.set_ylim(0.0, 1.05); numeric_ticks(ax, x=False, n=6)
    h = [Line2D([], [], marker="s", ls="", ms=9, mfc=C_BLUE, mec=INK, label="Constant offset"),
         Line2D([], [], marker="s", ls="", ms=9, mfc=C_RED, mec=INK, label="SR-GP"),
         Line2D([], [], marker="X", ls="", ms=9, mfc=C_LIGHT_PINK, mec=INK, label="Row-calibrated, study median")]
    ax.legend(handles=h, loc="lower center", fontsize=9.4, handletextpad=0.2)


# ================================================================== Fig. 5
def f5a(ax):
    z = dml[dml.FE_DML.notna()].copy()
    short = {"Alkali catalyst (vs none)": "Alkali catalyst", "Organic (co)solvent (vs water)": "Organic co-solvent",
             "Temperature (+10 C)": "Temperature +10 °C", "log residence time (+1 ln-unit)": "ln(1 + t) + 1",
             "Solid loading (+5 wt%)": "Solid loading +5 wt%"}
    z["lab"] = z.treatment.map(short) + np.where(z.outcome == "Y_oil", ", biocrude", ", char")
    z = z.iloc[::-1].reset_index(drop=True)
    for i, r in z.iterrows():
        ax.errorbar(r.naive_DML, i + 0.17, xerr=1.96 * r.naive_DML_se, fmt="o", ms=6.5, color=C_PINK, mec=INK, mew=0.8,
                    ecolor=C_RED, elinewidth=1.2, capsize=2.5, label="Pooled DML" if i == 0 else None)
        ax.errorbar(r.FE_DML, i - 0.17, xerr=1.96 * r.FE_DML_se, fmt="s", ms=6.5, color=C_BLUE, mec=INK, mew=1.0,
                    ecolor=INK, elinewidth=1.2, capsize=2.5, label="Within-study FE-DML" if i == 0 else None)
    ax.axvline(0, color=INK, lw=1.0, ls=":")
    ax.set_yticks(range(len(z))); ax.set_yticklabels(z.lab, fontsize=10)
    style_axis(ax, xlabel="Adjusted yield contrast (pp)", tick_size=10.5)
    ax.set_yticklabels(z.lab, fontsize=10)
    numeric_ticks(ax, y=False, n=6)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=10)


def f5b(ax):
    rng = np.random.default_rng(7)
    for j, (t, col, mk) in enumerate([("Y_oil", C_BLUE, "o"), ("Y_char", C_RED, "s")]):
        c = pd.read_csv(RES / f"04_cate_{t}_Alkali.csv")
        c["q"] = pd.qcut(c.Ash, 4, labels=False)
        m, lo, hi = [], [], []
        studies = c.study_id.unique()
        for qq in range(4):
            m.append(c[c.q == qq].tau.mean())
            bs = []
            for _ in range(300):
                sel = rng.choice(studies, len(studies), replace=True)
                zz = c.set_index("study_id").loc[sel].reset_index()
                bs.append(zz[zz.q == qq].tau.mean())
            lo.append(np.nanpercentile(bs, 2.5)); hi.append(np.nanpercentile(bs, 97.5))
        x = np.arange(4) + (j - 0.5) * 0.18
        ax.errorbar(x, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt=mk + "-", ms=8, color=col, mec=INK, mew=0.9,
                    ecolor=INK, capsize=3.5, lw=1.7, label="Biocrude" if t == "Y_oil" else "Char")
    ax.axhline(0, color=INK, lw=1.0, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    style_axis(ax, xlabel="Feedstock ash quartile", ylabel="Alkali contrast (pp)")
    numeric_ticks(ax, x=False, n=5)
    ax.legend(loc="lower right", fontsize=10)


def f5c(ax):
    pc = json.load(open(RES / "13_lab_effect_phase_corr.json"))
    M = np.array(pc["corr_mean"]); Sd = np.array(pc["corr_sd"])
    labs = ["Biocrude", "Char", "Aqueous", "Gas"]; n = len(labs)
    Mm = np.where(np.tril(np.ones((n, n)), -1) == 1, M, np.nan)
    im = ax.imshow(Mm, cmap=DIV, vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(n):
        for j in range(i):
            ax.text(j, i, f"{M[i, j]:.2f}\n±{Sd[i, j]:.2f}", ha="center", va="center", fontsize=10, color=INK)
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=30, ha="right"); ax.set_yticks(range(n)); ax.set_yticklabels(labs)
    style_axis(ax)
    ax.set_xticklabels(labs, rotation=30, ha="right")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    cax = ax.inset_axes([0.30, 0.9, 0.5, 0.035])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.6, 0, 0.6])
    cb.set_label("Fitted phase correlation", fontsize=9.0, labelpad=2); cb.ax.tick_params(labelsize=8.8, length=3, width=0.8)


def f5d(ax):
    rows = [("all", "All"), ("forced", "Gap <1 pp"), ("open", "Gap ≥3 pp or\nmissing phases"), ("oil_char_only_studies", "Oil + char only")]
    y = np.arange(len(rows))[::-1]
    for yi, (key, lb) in zip(y, rows):
        c = clo[key]
        ax.errorbar(c["r"], yi, xerr=[[c["r"] - c["lo"]], [c["hi"] - c["r"]]], fmt="o", ms=8, color=C_RED if key == "oil_char_only_studies" else C_BLUE,
                    mec=INK, mew=0.9, ecolor=INK, elinewidth=1.1, capsize=3)
    rn = clo["closure_null_projection"]["r_oil_char"]
    ax.axvline(rn, color=INK, lw=1.1, ls="--"); ax.text(rn + 0.03, y.max() + 0.62, "Closure-only expectation", ha="left", va="center", fontsize=8.9, color=INK)
    ax.axvline(0, color=INK, lw=0.9, ls=":")
    pn = clo["permutation_null_95"]
    ax.axvspan(pn[0], pn[1], color=C_GREEN, alpha=0.45, lw=0)
    for yi, (key, lb) in zip(y, rows):
        ax.text(0.21, yi, f"{lb}\n{clo[key]['n_studies']} studies", va="center", ha="left", fontsize=8.6, color=INK, linespacing=1.1)
    ax.set_yticks([])
    style_axis(ax, xlabel="Oil–char study-effect r")
    ax.spines["left"].set_visible(False); ax.tick_params(left=False)
    ax.set_xlim(-0.75, 0.85); ax.set_ylim(-0.6, len(rows) - 0.1); ax.set_xticks([-0.6, -0.3, 0.0, 0.3])


def f5e(ax):
    mo = pd.read_csv(RES / "13_mo_srgp_summary.csv")
    groups = [("Y_oil", "MO char->oil", "Biocrude"), ("Y_char", "MO oil->char", "Char")]
    x = np.arange(2); w = 0.25
    vals = {"No anchors": [], "Other phase": [], "Same phase": []}
    for t, cross, _ in groups:
        z = mo[(mo.phase == t) & (mo.k == 5)].set_index("model").R2
        vals["No anchors"].append(z["base"]); vals["Other phase"].append(z[cross]); vals["Same phase"].append(z["indep SR-GP"])
    for j, (nm, col) in enumerate(zip(vals, [C_LIGHT_BLUE, C_GREEN, C_RED])):
        ax.bar(x + (j - 1) * w, vals[nm], w, color=col, edgecolor=INK, lw=0.9, label=nm)
    ax.set_xticks(x); ax.set_xticklabels([gl for _, _, gl in groups])
    style_axis(ax, ylabel=f"Query {Q2}")
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", fontsize=9.0, labelspacing=0.25, handlelength=1.0, borderaxespad=0.1)


# ================================================================== supplementary
def _bench(ax, t):
    z = bench[(bench.target == t) & ~bench.model.str.startswith("Stacking")]
    g = z[z.split == "group"].set_index("model"); r = z[z.split == "random"].set_index("model")
    order = g.sort_values("R2").index.tolist()
    for i, m in enumerate(order):
        ax.plot([g.loc[m, "R2"], r.loc[m, "R2"]], [i, i], color=INK, lw=0.9, zorder=1)
        ax.scatter(r.loc[m, "R2"], i, s=46, color=ROLE["random"], edgecolor=INK, lw=0.7, zorder=3)
        ax.scatter(g.loc[m, "R2"], i, s=46, color=ROLE["group"], edgecolor=INK, lw=0.7, zorder=3, marker="s")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.axvline(0, color=INK, lw=0.9, ls=":")
    phase = "Biocrude" if t == "Y_oil" else "Char"
    style_axis(ax, xlabel=f"{phase} yield {R2}", tick_size=10.5)
    ax.set_yticklabels(order)
    ax.set_xlim(-2.08, 1.0); ax.set_xticks([-2.0, -1.0, 0, 1.0])
    h = [Line2D([], [], ls="", marker="o", ms=7, mfc=ROLE["random"], mec=INK, label="Random rows"),
         Line2D([], [], ls="", marker="s", ms=7, mfc=ROLE["group"], mec=INK, label="Held-out studies")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1), ncol=2, fontsize=9.5,
              handlelength=.8,handletextpad=.4,columnspacing=1.0)


def fs1a(ax): _bench(ax, "Y_oil")
def fs1b(ax): _bench(ax, "Y_char")


def fs2a(ax):
    cat = pd.read_csv(RES / "09_srgp_summary_Y_oil_Y_char_Y_aq_Y_gas_ER_oil_HHV_oil_catboost.csv")
    for t, mk in [("Y_oil", "o"), ("Y_char", "s")]:
        for df_, col, ls, lb in [(sr, C_BLUE, "--", "LightGBM base"), (cat, C_RED, "-", "CatBoost base")]:
            z = df_[(df_.target == t) & (df_.model == "SR-GP")].sort_values("k")
            ax.plot(z.k, z.R2, color=col, ls=ls, marker=mk, ms=8, lw=1.9, mec=INK, mew=0.9,
                    label=f"{lb}, {'biocrude' if t == 'Y_oil' else 'char'}")
    style_axis(ax, xlabel="Anchor runs in the new study", ylabel=f"SR-GP {Q2}")
    ax.set_xticks([1, 3, 5]); ax.set_ylim(0.45, 0.85); numeric_ticks(ax, x=False, n=5)
    ax.legend(loc="lower right", fontsize=10.1, labelspacing=0.3, handlelength=1.8)


# ================================================================== extended panels (2026-11-13)
P20 = json.loads((RES / "20_panel_support.json").read_text())
sizes = pd.read_csv(RES / "20_study_sizes.csv"); combos = pd.read_csv(RES / "20_phase_combos.csv")
pse = pd.read_csv(RES / "20_per_study_error_split.csv"); gain = pd.read_csv(RES / "20_rmse_gain_ci.csv")
four = pd.read_csv(RES / "20_fourphase_rmse.csv"); adapt = pd.read_csv(RES / "20_width_adaptivity.csv")
actps = pd.read_csv(RES / "20_active_per_study.csv"); exst = pd.read_csv(RES / "20_example_studies.csv")
csum = pd.read_csv(RES / "20_closure_sums.csv")
seff = {t: pd.read_csv(RES / f"10_study_effects_{t}.csv") for t in ["Y_oil", "Y_char"]}
phase_eff = pd.read_csv(RES / "15_study_effects_by_phase.csv")
SM = dict(tick_size=10, label_size=11.5)      # compact panels in three-column rows
WHITE_RED = LinearSegmentedColormap.from_list("white_rose", ["#FFFFFF", C_RED])


def g1_size(ax):
    s = sizes.sort_values("n", ascending=False).reset_index(drop=True)
    for sub, col in SUB_COL.items():
        m = s.subtype == sub
        ax.bar(np.where(m)[0] + 1, s.n[m], 1.0, color=col, lw=0)
    a = P20["study_size"]
    ax.text(0.97, 0.95, f"median {a['median']:.0f} records\ntop 20% of studies:\n{a['share_top20pct_studies']:.0%} of records",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.2, color=INK)
    style_axis(ax, xlabel="Source study rank", ylabel="Records", **SM)
    ax.set_xlim(0, len(s) + 1); numeric_ticks(ax, n=4)


def g1_combo(ax):
    lab = {"oil+char": "Oil + char", "oil+char+aq+gas": "All four", "oil": "Oil only", "char": "Char only",
           "none": "Other responses", "oil+char+gas": "Oil + char + gas", "oil+char+aq": "Oil + char + aq."}
    z = combos[combos.combo.isin(lab)].sort_values("records", ascending=True)
    y = np.arange(len(z))
    ax.barh(y, z.records, 0.68, color=C_LIGHT_BLUE, edgecolor=INK, lw=0.7)
    for yi, (rec, st) in enumerate(zip(z.records, z.studies)):
        ax.text(rec + 15, yi, f"{st}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([lab[c] for c in z.combo])
    style_axis(ax, xlabel="Records", **SM)
    ax.set_yticklabels([lab[c] for c in z.combo], fontsize=9.5)
    ax.set_xlim(0, z.records.max() * 1.18); numeric_ticks(ax, y=False, n=4)


def g1_offsets(ax):
    e = seff["Y_oil"].copy()
    keep = {"acetone": "Acetone", "DCM": "DCM", "filtration_or_other": "Filtration", "unreported": "Unreported"}
    e["grp"] = e.extraction.map(keep).fillna("Other solvent")
    order = ["Acetone", "DCM", "Other solvent", "Filtration", "Unreported"]
    data = [e.u[e.grp == g_].values for g_ in order]
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.58, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.3), whiskerprops=dict(color=INK), capprops=dict(color=INK))
    for b in bp["boxes"]: b.set_facecolor(C_PALE_BLUE); b.set_edgecolor(INK)
    rng = np.random.default_rng(3)
    for i, v in enumerate(data):
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=9, color=C_BLUE, edgecolor=INK, lw=0.3, zorder=3)
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    style_axis(ax, ylabel="Study offset (pp)", **SM)
    ax.set_xticklabels(order, fontsize=9, rotation=35, ha="right", rotation_mode="anchor")
    numeric_ticks(ax, x=False, n=5)


def g1_perstudy(ax):
    for t, col, mk, lb in [("Y_oil", C_BLUE, "o", "Biocrude"), ("Y_char", C_RED, "s", "Char")]:
        z = pse[pse.target == t]
        ax.scatter(z.rmse_random, z.rmse_group, s=16, marker=mk, facecolor="none", edgecolor=col, lw=1.0, label=lb)
    lim = [0, 45]; ax.plot(lim, lim, color=INK, lw=1.0, ls="--")
    a, b = P20["per_study_error_Y_oil"], P20["per_study_error_Y_char"]
    ax.text(0.97, 0.05, f"held-out higher in\n{a['share_studies_group_worse']:.0%} / {b['share_studies_group_worse']:.0%} of studies",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    style_axis(ax, xlabel="Random-row study RMSE (pp)", ylabel="Held-out study RMSE (pp)", **SM)
    numeric_ticks(ax, n=4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)


def g1_bench(ax):
    z = bench[~bench.model.str.startswith("Stacking")]; lo = -0.6
    for t, mk, col, lb in [("Y_oil", "o", C_BLUE, "Biocrude"), ("Y_char", "s", C_RED, "Char")]:
        zz = z[z.target == t]
        g_ = zz[zz.split == "group"].set_index("model").R2; r = zz[zz.split == "random"].set_index("model").R2.reindex(g_.index)
        inside = g_ >= lo
        ax.scatter(r[inside], g_[inside], s=34, marker=mk, color=col, edgecolor=INK, lw=0.7, label=lb, zorder=3)
        ax.scatter(r[~inside], np.full((~inside).sum(), lo + 0.04), s=34, marker="v", color=col, edgecolor=INK, lw=0.7, zorder=3)
    ax.plot([lo, 1], [lo, 1], color=INK, lw=1.0, ls="--"); ax.axhline(0, color=INK, lw=0.8, ls=":")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(lo, 1.0)
    style_axis(ax, xlabel=f"Random-row {R2}", ylabel=f"Held-out-study {R2}", **SM)
    numeric_ticks(ax, n=4)
    ax.legend(loc="upper left", fontsize=9, handletextpad=0.1, borderaxespad=0.1)


# ------------------------------------------------------------------ Fig. 2 additions
def g2_theta(ax):
    off = {"Y_oil": (0.12, 0.022), "Y_char": (0.1, -0.035), "Y_aq": (-0.62, -0.04), "Y_gas": (0.12, 0.02), "ER_oil": (0.16, -0.03), "HHV_oil": (-0.2, 0.03)}
    for t in ORDER:
        th = theta[t]
        l, r = th.full_l.median(), th.full_rho.median()
        ax.errorbar(l, r, xerr=[[l - th.full_l.min()], [th.full_l.max() - l]], yerr=[[r - th.full_rho.min()], [th.full_rho.max() - r]],
                    fmt="o", ms=7, color=C_RED, mec=INK, mew=0.8, ecolor=INK, elinewidth=0.6, capsize=0, zorder=3)
        ax.text(l + off[t][0], r + off[t][1], SHORT[t], fontsize=8.8, color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6))
    style_axis(ax, xlabel="Length scale $\\ell$", ylabel="Mismatch correlation $\\rho$", **SM)
    ax.set_xlim(0.9, 5.3); ax.set_ylim(0.55, 0.95); numeric_ticks(ax, n=4)


ABL = [("offset", C_BLUE, "o", "--", "Constant offset"), ("BLUP(fixed)", C_PALE_GREEN, "s", ":", "Fixed shrinkage"),
       ("kernel(fixed)", C_GREEN, "D", "-.", "Fixed kernel"), ("SR-GP(no local)", C_PINK, "v", "--", "Intercept only"),
       ("SR-GP(no intercept)", C_LIGHT_BLUE, "P", ":", "Local only"), ("SR-GP", C_RED, "^", "-", "SR-GP")]


def g2_abl(ax, t="Y_oil", legend=False):
    for m, col, mk, ls, lb in ABL:
        z = sr[(sr.target == t) & (sr.model == m)].sort_values("k")
        ax.plot(z.k, z.R2, color=col, marker=mk, ls=ls, lw=1.8, ms=6.5, mec=INK, mew=0.7, label=lb)
    ax.set_xticks([1, 3, 5])
    style_axis(ax, xlabel="Anchor runs", ylabel=f"{'Biocrude' if t == 'Y_oil' else 'Char'} {Q2}", **SM)
    numeric_ticks(ax, x=False, n=4)
    if legend:
        ax.legend(loc="lower right", fontsize=8.2, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)


def g2_case(ax, name="median", legend=False):
    z = exst[exst.case == name]; q_ = ~z.anchor
    rm = lambda c: float(np.sqrt(np.mean((z.y[q_] - z[c][q_]) ** 2)))
    for c, col, mk, lb, fill in [("pooled", C_GREEN, "s", "Pooled", False), ("offset", C_BLUE, "o", "Constant offset", True), ("srgp", C_RED, "^", "SR-GP", True)]:
        ax.scatter(z.y[q_], z[c][q_], s=20, marker=mk, facecolor=col if fill else "none", edgecolor=INK if fill else col, lw=0.5 if fill else 1.0,
                   label=f"{lb}" if legend else None, zorder=3 if c == "srgp" else 2)
    ax.scatter(z.y[z.anchor], z.srgp[z.anchor], s=95, marker="*", facecolor="white", edgecolor=INK, lw=0.9, zorder=4, label="Anchors" if legend else None)
    v = np.r_[z.y.values, z.pooled.values, z.offset.values, z.srgp.values]; lim = [max(0, v.min() - 4), v.max() + 4]
    ax.plot(lim, lim, color=INK, lw=1.0, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.97, 0.04, f"RMSE (pp)\npooled {rm('pooled'):.1f}\noffset {rm('offset'):.1f}\nSR-GP {rm('srgp'):.1f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.8, color=INK, linespacing=1.15)
    style_axis(ax, xlabel="Observed biocrude (%)", ylabel="Predicted biocrude (%)", **SM)
    numeric_ticks(ax, n=4)
    if legend:
        ax.legend(loc="upper left", fontsize=8.4, handletextpad=0.1, labelspacing=0.2, borderaxespad=0.1)


# ------------------------------------------------------------------ Fig. 3 additions
def g3_gain(ax):
    for t, col, mk, dx, lb in [("Y_oil", C_BLUE, "o", -0.1, "Biocrude"), ("Y_char", C_RED, "s", 0.1, "Char")]:
        z = gain[gain.target == t].sort_values("k")
        ax.errorbar(z.k + dx, z.gain, yerr=[z.gain - z.lo, z.hi - z.gain], fmt=mk + "-", ms=7, color=col, mec=INK, mew=0.8,
                    ecolor=INK, elinewidth=1.0, capsize=3, lw=1.5, label=lb)
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.set_xticks([1, 3, 5])
    style_axis(ax, xlabel="Anchor runs", ylabel="RMSE reduction (pp)", **SM)
    ax.set_ylim(-0.2, 2.8); numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="upper right", fontsize=9, handletextpad=0.2, borderaxespad=0.1)


def g3_study(ax, t="Y_oil", k=3):
    r = raw[t]; r = r[r.k == k]
    st = r.groupby(["study", "model"]).rmse_study.mean().unstack()
    col = C_BLUE if t == "Y_oil" else C_RED
    ax.scatter(st["offset"], st["SR-GP"], s=20, facecolor="none", edgecolor=col, lw=1.0)
    lim = [0, np.nanmax(st[["offset", "SR-GP"]].values) * 1.05]
    ax.plot(lim, lim, color=INK, lw=1.0, ls="--")
    nm = "biocrude" if t == "Y_oil" else "char"
    ax.text(0.05, 0.96, f"SR-GP lower in {np.mean(st['SR-GP'] < st['offset']):.0%}\nof {nm} studies", transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    style_axis(ax, xlabel="Offset RMSE (pp)", ylabel="SR-GP RMSE (pp)", **SM)
    numeric_ticks(ax, n=4)


def g3_win(ax):
    M = gain.pivot(index="target", columns="k", values="share_studies_srgp_lower").reindex(ORDER)
    im = ax.imshow(M.values, cmap=WHITE_RED, vmin=0.5, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.0%}", ha="center", va="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["1", "3", "5"]); ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[t] for t in ORDER])
    style_axis(ax, xlabel="Anchor runs", **SM)
    ax.set_yticklabels([SHORT[t] for t in ORDER])
    cax = ax.inset_axes([1.05, 0.0, 0.06, 1.0])
    cb = plt.colorbar(im, cax=cax, ticks=[0.5, 0.7, 0.9])
    cb.set_label("Studies with lower SR-GP RMSE", fontsize=8.8); cb.ax.tick_params(labelsize=8.5, length=3, width=0.8)


def g3_zero(ax):
    z = gap[gap.k == 1].set_index("target").reindex(ORDER); x = np.arange(len(ORDER)); w = 0.38
    ax.bar(x - w / 2, z.Q2_pooled_k0, w, color=C_RED, edgecolor=INK, lw=0.8, label="Pooled predictor")
    ax.bar(x + w / 2, z.Q2_GPB_best_k0, w, color=C_PINK, edgecolor=INK, lw=0.8, label="Better GPBoost mean")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in ORDER], rotation=35, ha="right", rotation_mode="anchor")
    style_axis(ax, ylabel=f"Zero-anchor {Q2}", **SM)
    ax.set_xticklabels([SHORT[t] for t in ORDER], rotation=35, ha="right", rotation_mode="anchor", fontsize=9.5)
    ax.set_ylim(-0.08, 0.62); numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="upper left", fontsize=8.8, handlelength=1.0, borderaxespad=0.1)


def g3_four(ax):
    for m, col, mk, ls, lb in [("base", C_GREEN, "s", ":", "Pooled, no anchors"), ("offset", C_BLUE, "o", "--", "Constant offsets"),
                               ("indep SR-GP", C_RED, "^", "-", "SR-GP per phase"), ("MO-SR-GP", C_PINK, "D", "-.", "Joint four-phase")]:
        z = four[four.model == m].sort_values("k")
        ax.plot(z.k, z.median_rmse, color=col, marker=mk, ls=ls, lw=1.8, ms=6.5, mec=INK, mew=0.7, label=lb)
    ax.set_xticks([1, 3, 5])
    style_axis(ax, xlabel="Anchor runs", ylabel="Four-phase RMSE (pp)", **SM)
    ax.set_ylim(5, 14.5); numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=8.6, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)


# ------------------------------------------------------------------ Fig. 4 additions
def g4_win(ax):
    for t, col, mk, lb in [("Y_oil", C_BLUE, "o", "Biocrude"), ("Y_char", C_RED, "s", "Char")]:
        z = act[(act.target == t) & (act.strategy == "IVR (active)")].sort_values("k")
        ax.plot(z.k, z.win_rate_vs_random, color=col, marker=mk, lw=1.8, ms=7, mec=INK, mew=0.8, label=lb)
    ax.axhline(0.5, color=INK, lw=0.9, ls=":")
    ax.set_xticks([1, 2, 3, 4, 5])
    style_axis(ax, xlabel="Anchor runs", ylabel="Studies favouring selection", **SM)
    ax.set_ylim(0.4, 0.85); numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="lower right", fontsize=9, handletextpad=0.2, borderaxespad=0.1)


def g4_diff(ax, k=3):
    for t, col, lb in [("Y_oil", C_BLUE, "Biocrude"), ("Y_char", C_RED, "Char")]:
        v = np.sort(actps[(actps.target == t) & (actps.k == k)].diff_random_minus_ivr.values)
        ax.plot(np.linspace(0, 1, len(v)), v, color=col, lw=2.0, label=f"{lb}, mean {v.mean():.2f}")
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    style_axis(ax, xlabel="Study quantile", ylabel="RMSE saved by selection (pp)", **SM)
    ax.set_ylim(-7, 9); numeric_ticks(ax, n=4)
    ax.legend(loc="upper left", fontsize=8.8, handlelength=1.4, borderaxespad=0.1)


def g4_adapt(ax):
    for t, mk, lb in [("Y_oil", "o", "biocrude"), ("Y_char", "s", "char")]:
        z = adapt[(adapt.target == t) & (adapt.model == "SR-GP")].sort_values("quintile")
        ax.plot(z.width, z.rmse, color=C_RED, marker=mk, lw=1.6, ms=6.5, mec=INK, mew=0.8, label=f"SR-GP, {lb}")
        o = adapt[(adapt.target == t) & (adapt.model == "offset")]
        ax.scatter(o.width, o.rmse, s=62, marker=mk, color=C_BLUE, edgecolor=INK, lw=0.8, zorder=3, label=f"Offset, {lb}")
    style_axis(ax, xlabel="Mean interval width (pp)", ylabel="Query RMSE (pp)", **SM)
    ax.set_ylim(5.5, 12.5); numeric_ticks(ax, n=4)
    ax.legend(loc="lower right", fontsize=8.4, handletextpad=0.2, labelspacing=0.2, borderaxespad=0.1)


def g4_zero(ax):
    meth = [("naive_row", C_PINK, "Row-calibrated"), ("study", C_BLUE, "Study-held-out"), ("mondrian_subtype", C_GREEN, "By process subtype")]
    x = np.arange(2); w = 0.26
    for j, (m, col, lb) in enumerate(meth):
        v = [conf[(conf.target == t) & (conf.method == m) & (conf.k == 0)].coverage.values[0] for t in ["Y_oil", "Y_char"]]
        ax.bar(x + (j - 1) * w, v, w, color=col, edgecolor=INK, lw=0.8, label=lb)
    ax.axhline(0.9, color=INK, ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(["Biocrude", "Char"])
    style_axis(ax, ylabel="Zero-anchor coverage", **SM)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.3, 0.6, 0.9])
    ax.legend(loc="upper left", fontsize=8.6, handlelength=1.0, labelspacing=0.2, borderaxespad=0.1)


# ------------------------------------------------------------------ Fig. 5 additions
def g5_ref(ax, labels=False):
    z = dml[dml.FE_DML.notna()].copy().iloc[::-1].reset_index(drop=True)
    for i, r in z.iterrows():
        ax.errorbar(r.placebo_mean, i + 0.2, xerr=1.96 * r.placebo_sd, fmt="o", ms=5.5, color=C_PALE_GREEN, mec=INK, mew=0.7, ecolor=INK,
                    elinewidth=0.9, capsize=2, label="Permuted" if i == 0 else None)
        ax.errorbar(r.subsample_mean, i - 0.2, xerr=1.96 * r.subsample_sd, fmt="D", ms=5.5, color=C_LIGHT_BLUE, mec=INK, mew=0.7, ecolor=INK,
                    elinewidth=0.9, capsize=2, label="Subsampled" if i == 0 else None)
        ax.scatter(r.FE_DML, i, s=34, marker="s", color=C_BLUE, edgecolor=INK, lw=0.8, zorder=4, label="Estimate" if i == 0 else None)
        ax.text(1.02, i, f"{int(r.n_identifying_studies)}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9, color=INK)
    ax.text(1.02, len(z) - 0.25, "studies", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8.6, color=INK)
    ax.axvline(0, color=INK, lw=1.0, ls=":")
    short = {"Alkali catalyst (vs none)": "Alkali", "Organic (co)solvent (vs water)": "Organic", "Temperature (+10 C)": "T +10 °C",
             "log residence time (+1 ln-unit)": "ln(1 + t)", "Solid loading (+5 wt%)": "Loading"}
    labs = (z.treatment.map(short) + np.where(z.outcome == "Y_oil", ", oil", ", char")).tolist()
    ax.set_yticks(range(len(z))); ax.set_yticklabels(labs if labels else [""] * len(z))
    style_axis(ax, xlabel="Contrast (pp)", **SM)
    ax.set_yticklabels(labs if labels else [""] * len(z), fontsize=9)
    ax.set_ylim(-0.6, len(z) - 0.4); ax.set_xlim(-9, 17); numeric_ticks(ax, y=False, n=5)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), ncol=1, fontsize=8.4, handletextpad=0.2, labelspacing=0.25, borderaxespad=0.2)


def _moderator(ax, treat, column, edges, ticklabels, xlabel, ylabel, legend=False):
    rng = np.random.default_rng(11)
    for j, (t, col, mk) in enumerate([("Y_oil", C_BLUE, "o"), ("Y_char", C_RED, "s")]):
        c = pd.read_csv(RES / f"04_cate_{t}_{treat}.csv")
        c["q"] = pd.cut(c[column], edges, labels=False) if not isinstance(edges, int) else pd.qcut(c[column], edges, labels=False, duplicates="drop")
        nb = int(c.q.max()) + 1; m, lo, hi = [], [], []
        studies = c.study_id.unique(); ci = c.set_index("study_id")
        for qq in range(nb):
            m.append(c[c.q == qq].tau.mean()); bs = []
            for _ in range(300):
                zz = ci.loc[rng.choice(studies, len(studies), replace=True)]
                bs.append(zz[zz.q == qq].tau.mean())
            lo.append(np.nanpercentile(bs, 2.5)); hi.append(np.nanpercentile(bs, 97.5))
        x = np.arange(nb) + (j - 0.5) * 0.16
        ax.errorbar(x, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt=mk + "-", ms=6.5, color=col, mec=INK, mew=0.8, ecolor=INK,
                    capsize=3, lw=1.5, label="Biocrude" if t == "Y_oil" else "Char")
    ax.axhline(0, color=INK, lw=1.0, ls=":")
    ax.set_xticks(range(len(ticklabels))); ax.set_xticklabels(ticklabels)
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, **SM)
    ax.set_xticklabels(ticklabels, fontsize=9.2)
    numeric_ticks(ax, x=False, n=5)
    if legend: ax.legend(loc="lower right", fontsize=9, handletextpad=0.2, borderaxespad=0.1)


def g5_ash(ax): _moderator(ax, "Alkali", "Ash", 4, ["Q1", "Q2", "Q3", "Q4"], "Feedstock ash quartile", "Alkali contrast (pp)", legend=True)
def g5_temp(ax): _moderator(ax, "Temperature", "T", [0, 250, 300, 350, 600], ["≤250", "250–300", "300–350", ">350"], "Temperature (°C)", "Contrast per +10 °C (pp)")
def g5_lignin(ax): _moderator(ax, "Organic", "Lignin", 3, ["Low", "Middle", "High"], "Feedstock lignin tercile", "Organic-medium contrast (pp)")


def g5_scatter(ax):
    b = phase_eff.dropna(subset=["Y_oil", "Y_char"]).copy()
    grp = np.where(~b.has_aq_gas, "Oil + char only", np.where(b.balance == "forced", "Sum within 1 pp", "Other"))
    for g_, col, mk in [("Sum within 1 pp", C_BLUE, "o"), ("Other", C_PALE_GREEN, "D"), ("Oil + char only", C_RED, "s")]:
        m = grp == g_
        ax.scatter(b.Y_oil[m], b.Y_char[m], s=22, marker=mk, color=col, edgecolor=INK, lw=0.5, label=g_)
    k1, k0 = np.polyfit(b.Y_oil, b.Y_char, 1); xx = np.array([b.Y_oil.min(), b.Y_oil.max()])
    ax.plot(xx, k0 + k1 * xx, color=INK, lw=1.2, ls="--")
    ax.axhline(0, color=INK, lw=0.7, ls=":"); ax.axvline(0, color=INK, lw=0.7, ls=":")
    ax.text(0.97, 0.95, f"r = {clo['all']['r']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK)
    style_axis(ax, xlabel="Biocrude study effect (pp)", ylabel="Char study effect (pp)", **SM)
    numeric_ticks(ax, n=4)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.12, 1.0), ncol=2, fontsize=8.4, handletextpad=0.1, columnspacing=0.5,
              labelspacing=0.2, borderaxespad=0.1)


def g5_sums(ax):
    dev = (csum.sum4 - 100).abs()
    edges = [0, 1, 5, 20, np.inf]; labs = ["<1", "1–5", "5–20", ">20"]
    cnt = [int(((dev >= a) & (dev < b)).sum()) for a, b in zip(edges[:-1], edges[1:])]
    ax.bar(range(4), cnt, 0.7, color=C_LIGHT_BLUE, edgecolor=INK, lw=0.8)
    for i, c in enumerate(cnt): ax.text(i, c + 12, f"{c / len(dev):.0%}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs)
    style_axis(ax, xlabel="|Four-phase sum − 100| (pp)", ylabel="Records", **SM)
    ax.set_ylim(0, max(cnt) * 1.15); numeric_ticks(ax, x=False, n=4)



# ================================================================== composition engine
GS_DEFAULT = dict(left=0.1, right=0.985, top=0.91, bottom=0.07, wspace=0.75, hspace=0.6)


def measure_footprints(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return {key: ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for key, ax in axes.items()}


def align_mosaic_footprints(fig, axes, mosaic, omit=(), height_ratios=None, width_ratios=None):
    """Fit full panel footprints to compact cells, including labels and legends.

    Adapted from the reference project's figkit.py full-footprint fitting
    principle. Recalculated from rendered artists on every figure run.
    """
    from matplotlib.transforms import Bbox
    array = np.asarray(mosaic); nr, nc = array.shape
    hr = np.asarray(height_ratios or [1]*nr, dtype=float)
    wr = np.asarray(width_ratios or [1]*nc, dtype=float)
    left, right, bottom, top = .045, .985, .025, .972
    gx = .045 if nc > 1 else 0
    # Mosaic columns may be narrow subdivisions for mixed panel spans.
    if nc > 2: gx = .014
    gy = .035 if nr > 1 else 0
    cw = wr/wr.sum()*(right-left-gx*(nc-1))
    rh = hr/hr.sum()*(top-bottom-gy*(nr-1))
    xs = left + np.r_[0,np.cumsum(cw+gx)[:-1]]
    yt = top - np.r_[0,np.cumsum(rh+gy)[:-1]]
    targets = {}
    for key in axes:
        cell = np.argwhere(array==key); r0,c0 = cell.min(axis=0); r1,c1 = cell.max(axis=0)
        targets[key] = Bbox.from_extents(xs[c0],yt[r1]-rh[r1],xs[c1]+cw[c1],yt[r0])
    for _ in range(18):
        footprints = measure_footprints(fig, axes)
        errors = []
        for key, ax in axes.items():
            box, target = footprints[key], targets[key]
            errors.append(np.max(np.abs(np.asarray(box.extents)-target.extents)))
            p = ax.get_position(original=True)
            sx,sy = target.width/box.width,target.height/box.height
            ax.set_position([target.x0+(p.x0-box.x0)*sx,target.y0+(p.y0-box.y0)*sy,p.width*sx,p.height*sy])
        if max(errors)<.00035: break
    footprints = measure_footprints(fig, axes)
    fig._alignment_report = {
        key: {"target":list(targets[key].extents), "actual":list(b.extents),
              "max_error_pt":float(np.max(np.abs(np.asarray(b.extents)-targets[key].extents)*np.tile(fig.get_size_inches()*72,2)))}
        for key,b in footprints.items() if key not in omit}
    return footprints


def compose(stem, mosaic, funcs, figsize, letters, singles, width_ratios=None, height_ratios=None, gs=None,
            standalone_funcs=None, omit_alignment=(), adjust=None, match=()):
    """Export a mixed-span composite and independently render each panel."""
    folder = FIG / stem; sub = folder / "subfigures"; sub.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    axd = fig.subplot_mosaic(mosaic, width_ratios=width_ratios, height_ratios=height_ratios,
                             gridspec_kw={**GS_DEFAULT, **(gs or {})})
    for key, f in funcs.items(): f(axd[key])
    for key, (dl, db, dr, dt) in (adjust or {}).items():   # room for long category labels
        b0 = axd[key].get_position(); axd[key].set_position([b0.x0 + dl, b0.y0 + db, b0.width - dl + dr, b0.height - db + dt])
    for old in sub.glob(f"{stem}_*"): old.unlink()
    normal_text(fig)
    footprint = align_mosaic_footprints(fig, axd, mosaic, omit=omit_alignment,
                                       height_ratios=height_ratios,width_ratios=width_ratios)
    for key, ref in match:   # panels that share category rows use the same vertical extent
        p_, r_ = axd[key].get_position(), axd[ref].get_position()
        axd[key].set_position([p_.x0, r_.y0, p_.width, r_.height])
    if match: footprint = measure_footprints(fig, axd)
    place = {}
    for letter, key in letters.items():
        tb = footprint[key]
        row = next(i for i, slots in enumerate(mosaic) if key in slots)
        place[letter] = [tb.x0 - 0.014, tb.y1, row]
    for letter, (x, top, row) in place.items():
        y = max(t for _, t, r in place.values() if r == row)
        fig.text(x, y + 0.004, letter, ha="left", va="top", **LETTER)
    audit = audit_colors(fig)
    audit['regular_text_artists'] = audit_text_weights(fig)
    audit['panel_footprints'] = {key: list(b.bounds) for key, b in footprint.items()}
    audit['alignment'] = fig._alignment_report
    save_figure_with_subfigures(fig, folder, stem, dpi=600, close=True, export_panels=False)
    audit['standalone_regular_text_artists'] = {}
    for letter, (keys, size) in singles.items():
        fg = plt.figure(figsize=size)
        ad = fg.subplot_mosaic([keys], gridspec_kw=dict(wspace=0.5))
        for kk in keys: (standalone_funcs or {}).get(kk, funcs[kk])(ad[kk])
        if stem=="Fig02_residual_calibration" and letter=="a":
            ad["a"].set_position([.025,.05,.95,.90])
        normal_text(fg); audit_colors(fg)
        audit['standalone_regular_text_artists'][letter] = audit_text_weights(fg)
        fg.savefig(sub / f"{stem}_{letter}.png", dpi=600, bbox_inches="tight", facecolor="white")
        fg.savefig(sub / f"{stem}_{letter}.pdf", bbox_inches="tight", facecolor="white",
                   metadata={"Creator":"CHONG LIU","Author":"CHONG LIU"})
        plt.close(fg)
    (sub / "README.md").write_text("Each panel in this folder is redrawn independently on its own canvas by "
                                   "scripts/16_publication_figures.py; none is cropped from the combined figure. "
                                   "Panel letters follow the combined figure.\n", encoding="utf-8")
    return audit


def g1_lolli(ax):
    for i, t in enumerate(ORDER[::-1]):
        r = q[(q.target == t) & (q.split == "random")].R2.values[0]
        g_ = q[(q.target == t) & (q.split == "group")].R2.values[0]
        ax.plot([g_, r], [i, i], color=INK, lw=1.2, zorder=1)
        ax.scatter(r, i, s=58, color=ROLE["random"], edgecolor=INK, lw=0.8, zorder=3, label="Random rows" if i == 0 else None)
        ax.scatter(g_, i, s=58, color=ROLE["group"], edgecolor=INK, lw=0.8, marker="s", zorder=3, label="Held-out studies" if i == 0 else None)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[t] for t in ORDER[::-1]])
    ax.axvline(0, color=INK, lw=0.9, ls=":")
    style_axis(ax, xlabel=f"Cross-validated {R2}", **SM)
    ax.set_yticklabels([SHORT[t] for t in ORDER[::-1]])
    ax.set_xlim(-0.2, 1.0); ax.set_xticks([0, 0.5, 1.0])
    ax.legend(loc="lower left", bbox_to_anchor=(-0.2, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.5, fontsize=8.8, borderaxespad=0.1)



def g1_budget(ax):
    seg = [("between_study", C_RED, "Between studies"), ("within_transferable(new study)", C_GREEN, "Within, transferable"),
           ("within_study_specific", C_BLUE, "Within, random-row gain"), ("within_unexplained", C_PINK, "Within, unexplained")]
    for y, t in zip([1, 0], ["Y_oil", "Y_char"]):
        left = 0
        for key, col, lb in seg:
            v = lab[t]["budget"][key]
            ax.barh(y, v, 0.55, left=left, color=col, edgecolor=INK, lw=0.8, label=lb if t == "Y_oil" else None)
            if v > 0.12: ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=9.5, color=INK)
            left += v
    ax.set_yticks([1, 0]); ax.set_yticklabels(["Biocrude", "Char"])
    style_axis(ax, xlabel="Share of yield variance", **SM)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.5); numeric_ticks(ax, y=False, n=5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.0), ncol=2, fontsize=8.4, handlelength=0.9, columnspacing=0.5,
              handletextpad=0.3, borderaxespad=0.1, labelspacing=0.25)


def g1_icc(ax):
    x = np.arange(2); w = 0.36
    for j, (key, col, lb) in enumerate([("ICC_raw", C_BLUE, "Observed yield"), ("ICC_resid", C_RED, "Held-out residual")]):
        vals = [lab[t][key] for t in ("Y_oil", "Y_char")]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, edgecolor=INK, lw=0.8, label=lb)
    ax.set_xticks(x); ax.set_xticklabels(["Biocrude", "Char"])
    style_axis(ax, ylabel="Intraclass correlation", **SM)
    ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", fontsize=8.4, handlelength=0.9, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)


def g3_matched(ax, t, legend):
    methods = [("offset", C_BLUE, "o", "--", "Constant offset"), ("GPB-RE", C_GREEN, "s", ":", "GPBoost + intercept"),
               ("GPB-RE+GP", C_PINK, "D", "-.", "GPBoost + local GP"), ("SR-GP", C_RED, "^", "-", "SR-GP")]
    for m, col, mk, ls, lb in methods:
        z = gpb[(gpb.target == t) & (gpb.model == m)].sort_values("k")
        ax.plot(z.k, z.R2, color=col, marker=mk, ls=ls, lw=1.8, ms=6.5, mec=INK, mew=0.7, label=lb)
    ax.set_xticks([1, 3, 5]); ax.set_ylim(0.3, 0.85)
    style_axis(ax, xlabel="Anchor runs", ylabel=f"{'Biocrude' if t == 'Y_oil' else 'Char'} query {Q2}", **SM)
    numeric_ticks(ax, x=False, n=5)
    if legend:
        ax.legend(loc="lower right", fontsize=8.3, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)


def g4_design(ax, t, legend):
    for stg, col, mk, ls, lb in [("random", C_BLUE, "o", "--", "Random"), ("max-min distance", C_GREEN, "D", ":", "Space-filling"),
                                 ("IVR (active)", C_RED, "^", "-", "Variance-selected")]:
        z = act[(act.target == t) & (act.strategy == stg)].sort_values("k")
        ax.plot(z.k, z.R2, color=col, ls=ls, lw=1.8, marker=mk, ms=6.5, mec=INK, mew=0.7, label=lb)
    r5 = act[(act.target == t) & (act.strategy == "random") & (act.k == 5)].R2.iloc[0]
    ax.axhline(r5, color=INK, lw=0.9, ls=(0, (2, 2)))
    style_axis(ax, xlabel="Anchor runs", ylabel=f"{'Biocrude' if t == 'Y_oil' else 'Char'} query {Q2}", **SM)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_ylim(0.46, 0.86); numeric_ticks(ax, x=False, n=5)
    if legend:
        ax.legend(loc="lower right", fontsize=8.6, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)


S3 = (3.9, 3.5)
FIGS = {
    "Fig01_nested_literature": dict(
        mosaic=[list("aaaaaaaabbbb"), list("ccccddddeeee"), list("fffffggghhhh"), list("iiiijjjjkkkk")],
        funcs={"a": f1a, "b": f1b, "c": g1_size, "d": g1_combo, "e": f1c, "f": g1_budget, "g": g1_icc, "h": g1_offsets,
               "i": g1_lolli, "j": g1_perstudy, "k": g1_bench},
        figsize=(9.4, 11.0), height_ratios=[1.3, 1, 1, 1.05], gs=dict(left=0.1, wspace=1.0, hspace=0.62, top=0.955, bottom=0.05),
        letters={c: c for c in "abcdefghijk"},
        singles={"a": (["a"], (6.2, 4.4)), "b": (["b"], (3.4, 3.6)), **{c: ([c], S3) for c in "cdefghijk"}},
        standalone_funcs={"b": lambda ax: f1b(ax, legend=True)}),
    "Fig02_residual_calibration": dict(
        mosaic=[["a", "a", "a"], ["b", "c", "d"], ["e", "f", "g"], ["h", "i", "j"]],
        funcs={"a": f2a, "b": f2b, "c": f2c, "d": f2d, "e": g2_theta, "f": lambda ax: g2_abl(ax, "Y_oil"), "g": lambda ax: g2_abl(ax, "Y_char", legend=True),
               "h": lambda ax: g2_case(ax, "lower quartile", legend=True), "i": lambda ax: g2_case(ax, "median"), "j": lambda ax: g2_case(ax, "upper quartile")},
        figsize=(9.4, 10.9), height_ratios=[0.82, 1.0, 1.0, 1.0], gs=dict(top=0.975, wspace=0.5, hspace=0.6, bottom=0.05),
        letters={c: c for c in "abcdefghij"},
        singles={"a": (["a"], (8.6, 2.6)), **{c: ([c], S3) for c in "bcdefghij"}},
        standalone_funcs={"b": lambda ax: _variogram(ax, "Y_oil", legend=True), "f": lambda ax: g2_abl(ax, "Y_oil", legend=True),
                          "i": lambda ax: g2_case(ax, "median", legend=True), "j": lambda ax: g2_case(ax, "upper quartile", legend=True)},
        omit_alignment=("a",)),
    "Fig03_few_shot_accuracy": dict(
        mosaic=[["a", "a", "b", "b", "c", "c"], ["d", "d", "e", "e", "f", "f"], ["g", "g", "h", "h", "i", "i"], ["j", "j", "j", "k", "k", "k"]],
        funcs={"a": lambda ax: g3_matched(ax, "Y_oil", True), "b": lambda ax: g3_matched(ax, "Y_char", False), "c": f3c, "d": g3_gain, "e": lambda ax: g3_study(ax, "Y_oil"), "f": lambda ax: g3_study(ax, "Y_char"),
               "g": g3_win, "h": g3_zero, "i": f3d, "j": f3e, "k": g3_four},
        figsize=(9.4, 11.0), gs=dict(wspace=1.0, hspace=0.66, top=0.955, bottom=0.05),
        letters={c: c for c in "abcdefghijk"},
        singles={c: ([c], S3) for c in "abcdefghijk"}, standalone_funcs={"b": lambda ax: g3_matched(ax, "Y_char", True)}),
    "Fig04_uncertainty_design": dict(
        mosaic=[["a", "b", "c"], ["d", "e", "f"], ["g", "h", "h"]],
        funcs={"a": lambda ax: g4_design(ax, "Y_oil", True), "b": lambda ax: g4_design(ax, "Y_char", False), "c": g4_win, "d": g4_diff, "e": lambda ax: f4c(ax, legend_columns=1), "f": g4_adapt, "g": g4_zero, "h": f4d},
        figsize=(9.4, 9.6), gs=dict(hspace=0.5, wspace=0.5, top=0.965, bottom=0.06),
        letters={c: c for c in "abcdefgh"},
        singles={c: ([c], S3) for c in "abcdefgh"},
        standalone_funcs={"b": lambda ax: g4_design(ax, "Y_char", True)}),
    "Fig05_contrasts_coupling": dict(
        mosaic=[list("aaaaaaaabbbb"), list("ccccddddeeee"), list("ffffgggghhhh"), list("iiiiijjjjjjj")],
        funcs={"a": f5a, "b": g5_ref, "c": g5_ash, "d": g5_temp, "e": g5_lignin, "f": g5_scatter, "g": f5c, "h": f5d, "i": g5_sums, "j": f5e},
        figsize=(9.4, 10.9), height_ratios=[1.35, 1.0, 1.0, 0.9], gs=dict(left=0.1, wspace=1.0, hspace=0.6, top=0.93, bottom=0.05),
        letters={c: c for c in "abcdefghij"},
        singles={"a": (["a"], (5.0, 5.2)), "b": (["b"], (4.2, 5.2)), **{c: ([c], S3) for c in "cdefghij"}},
        standalone_funcs={"b": lambda ax: g5_ref(ax, labels=True)},
        match=(("b", "a"),)),
    "FigS01_model_benchmark": dict(
        mosaic=[["a", "b"]], funcs={"a": fs1a, "b": fs1b}, figsize=(8.4, 8.6),
        letters={"a": "a", "b": "b"}, singles={"a": (["a"], (4.4, 8.0)), "b": (["b"], (4.4, 8.0))}),
    "FigS02_base_learner_sensitivity": dict(
        mosaic=[["a"]], funcs={"a": fs2a}, figsize=(5.4, 4.4), letters={}, singles={"a": (["a"], (5.4, 4.4))}),
}


def main(only=None):
    import nbformat
    for stem, spec in FIGS.items():
        if only and stem not in only: continue
        audit = compose(stem, **spec)
        (QA / f"{stem}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell("# Reproduce this figure\n\nRuns the publication plotting entry point using the accompanying numerical results."),
            nbformat.v4.new_code_cell("from pathlib import Path\nimport runpy, sys\nroots=[Path.cwd(),*Path.cwd().parents]\n"
                                      "script=next(p/'scripts'/'16_publication_figures.py' for p in roots if (p/'scripts'/'16_publication_figures.py').exists())\n"
                                      f"sys.argv=[str(script),'--only','{stem}']\nrunpy.run_path(str(script),run_name='__main__')")])
        nbformat.write(nb, FIG / stem / f"{stem}_jupyter_code.ipynb")
        print("Exported", stem, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--only", nargs="*")
    main(parser.parse_args().only)
