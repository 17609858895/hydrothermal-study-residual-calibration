"""Does the pool-then-localize advantage over GPBoost grow with zero-shot transferable signal?

Reads the 14_gpboost_raw_{target}.csv files (same anchor/query episodes for every model within a target) and,
per target and anchor budget k, reports
  Q2(model)          = 1 - sum SSE / sum SST over all episodes (as in 09/14)
  gap_GP             = Q2(SR-GP) - Q2(GPB-RE+GP)        (jointly fitted intercept + local GP)
  gap_best           = Q2(SR-GP) - max(Q2(GPB-RE), Q2(GPB-RE+GP))
  zs_pooled          = Q2 of the pooled LightGBM at k = 0 (zero-shot transferable signal)
  zs_loss            = zs_pooled - max zero-shot Q2 of the GPBoost fixed effects (signal the joint fit absorbed)
  gap_kernel         = Q2(SR-GP@GPB) - Q2(GPB-RE+GP): held-out-residual covariance vs GPBoost's jointly fitted
                       intercept + GP, both on GPBoost trees (trees fitted separately, so not a pure kernel swap)
with 95% study-bootstrap intervals for the gaps (studies resampled within target, 2000 draws).
Across targets: Spearman and Pearson correlation of gap with zs_pooled and with zs_loss (n = number of targets;
descriptive only - six targets cannot establish a trend).
Usage: 18_gpboost_gap_vs_transfer.py [fixed|tuned]. "tuned" uses 14_gpboost_raw_{t}_tuned.csv (target-specific
boosting rounds from 14b) where it exists and the 1500-round file otherwise; the file used is recorded.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
from common import RES, SEED

TGT = ["Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "HHV_oil"]
GPB = ["GPB-RE", "GPB-RE+GP"]
NB = 2000
rng = np.random.default_rng(SEED + 18)
MODE = sys.argv[1] if len(sys.argv) > 1 else "fixed"
SUF = "" if MODE == "fixed" else "_tuned"


def q2(z):
    return 1 - z.sse.sum() / z.sst.sum()


rows = []
for t in TGT:
    f = RES / f"14_gpboost_raw_{t}{SUF}.csv"
    if SUF and not f.exists():
        f = RES / f"14_gpboost_raw_{t}.csv"
    if not f.exists():
        print("missing", f); continue
    R = pd.read_csv(f)
    # per (study, k, model) sums -> fast bootstrap over studies
    A = R.groupby(["k", "study", "model"])[["sse", "sst"]].sum().reset_index()
    for k, Ak in A.groupby("k"):
        P = {m: g.set_index("study")[["sse", "sst"]] for m, g in Ak.groupby("model")}
        studies = P["SR-GP"].index.values
        P = {m: v.reindex(studies) for m, v in P.items()}

        def score(ix):
            s = {m: 1 - v.sse.values[ix].sum() / v.sst.values[ix].sum() for m, v in P.items()}
            return s
        full = score(np.arange(len(studies)))
        bs_gp, bs_best, bs_k = [], [], []
        for _ in range(NB):
            ix = rng.integers(0, len(studies), len(studies))
            s = score(ix)
            bs_gp.append(s["SR-GP"] - s["GPB-RE+GP"])
            bs_best.append(s["SR-GP"] - max(s[g] for g in GPB))
            bs_k.append(s["SR-GP@GPB"] - s["GPB-RE+GP"])
        zs_gpb = max(full[f"{g} (k=0)"] for g in GPB)
        rows.append({"target": t, "k": k, "file": f.name, "n_studies": len(studies),
                     "Q2_pooled_k0": full["pooled (k=0)"], "Q2_GPB_best_k0": zs_gpb,
                     "Q2_offset": full["offset"], "Q2_SRGP": full["SR-GP"],
                     "Q2_GPB_RE": full["GPB-RE"], "Q2_GPB_RE_GP": full["GPB-RE+GP"], "Q2_SRGP_at_GPB": full["SR-GP@GPB"],
                     "gap_GP": full["SR-GP"] - full["GPB-RE+GP"],
                     "gap_GP_lo": np.percentile(bs_gp, 2.5), "gap_GP_hi": np.percentile(bs_gp, 97.5),
                     "gap_best": full["SR-GP"] - max(full[g] for g in GPB),
                     "gap_best_lo": np.percentile(bs_best, 2.5), "gap_best_hi": np.percentile(bs_best, 97.5),
                     "gap_kernel": full["SR-GP@GPB"] - full["GPB-RE+GP"],
                     "gap_kernel_lo": np.percentile(bs_k, 2.5), "gap_kernel_hi": np.percentile(bs_k, 97.5),
                     "zs_loss": full["pooled (k=0)"] - zs_gpb})

if MODE == "tuned":   # model-level summary actually used for Table S5 / Fig. 3a,b
    S = []
    for t in TGT:
        f = RES / f"14_gpboost_summary_{t}_tuned.csv"
        if f.exists():
            S.append(pd.read_csv(f))
        else:
            z = pd.read_csv(RES / "14_gpboost_summary_Y_oil_Y_char.csv"); z = z[z.target == t].copy()
            z["n_rounds"] = 1500; S.append(z)
    pd.concat(S).to_csv(RES / "14_gpboost_summary_six_targets.csv", index=False)

T = pd.DataFrame(rows)
T.to_csv(RES / f"18_gpboost_gap_by_target_{MODE}.csv", index=False)
out = {"n_targets": int(T.target.nunique()), "by_k": {}}
for k, z in T.groupby("k"):
    o = {}
    for gap in ["gap_GP", "gap_best"]:
        for x in ["Q2_pooled_k0", "zs_loss"]:
            if len(z) >= 3:
                rs = stats.spearmanr(z[x], z[gap]); rp = stats.pearsonr(z[x], z[gap])
                o[f"{gap}~{x}"] = {"spearman": float(rs.statistic), "spearman_p": float(rs.pvalue),
                                   "pearson": float(rp.statistic), "pearson_p": float(rp.pvalue)}
    o["targets_SRGP_ahead_CI_excl_0"] = z.loc[z.gap_best_lo > 0, "target"].tolist()
    o["targets_GPB_ahead_CI_excl_0"] = z.loc[z.gap_best_hi < 0, "target"].tolist()
    o["kernel_positive_targets"] = int((z.gap_kernel > 0).sum())
    o["kernel_CI_excl_0_targets"] = z.loc[z.gap_kernel_lo > 0, "target"].tolist()
    out["by_k"][int(k)] = o
json.dump(out, open(RES / f"18_gpboost_gap_vs_transfer_{MODE}.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(T.drop(columns="file").round(3).to_string(index=False))
print(json.dumps(out, indent=1))
