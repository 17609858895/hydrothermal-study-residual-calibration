"""Conformal intervals for an UNSEEN study (outer study-grouped 5-fold).

calibration residuals come from inner CV on the outer-training studies:
  naive   : inner random K-fold OOF residuals (row exchangeability -- what most papers do)
  study   : inner study-grouped OOF residuals (study exchangeability)
  study-k : few-shot adapted residuals (k anchors per calibration study, offset-corrected), k=1,3,5
  mondrian: study-level, stratified by process subtype
Reports coverage, mean width, and the share of test studies whose own coverage < 0.8.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

ALPHA = 0.10
RNG = np.random.default_rng(SEED)
LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED, n_jobs=6)


def qhat(res, alpha=ALPHA):
    n = len(res)
    return np.quantile(res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def oof(X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]).predict(X[te])
    return p


def adapted_residuals(y, p, g, k, n_draw=10):
    res = []
    for s in np.unique(g):
        idx = np.where(g == s)[0]
        if len(idx) <= k + 1:
            continue
        for _ in range(n_draw):
            anc = RNG.choice(idx, k, replace=False); q = np.setdiff1d(idx, anc)
            res.append(np.abs(y[q] - (p[q] + np.mean(y[anc] - p[anc]))))
    return np.concatenate(res)


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values; g = d.study_id.values; sub = d.process_subtype.values
    rec = []
    for tr, te in splits(d, "group"):
        dtr = d.iloc[tr].reset_index(drop=True)
        p_in_g = oof(X[tr], y[tr], splits(dtr, "group", seed=SEED + 7))
        p_in_r = oof(X[tr], y[tr], splits(dtr, "random", seed=SEED + 7))
        m = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]); pt = m.predict(X[te])
        q = {"naive_row": qhat(np.abs(y[tr] - p_in_r)), "study": qhat(np.abs(y[tr] - p_in_g))}
        qm = {s_: qhat(np.abs(y[tr] - p_in_g)[sub[tr] == s_]) if np.sum(sub[tr] == s_) > 30 else q["study"]
              for s_ in np.unique(sub)}
        for i, j in enumerate(te):
            for nm in ["naive_row", "study"]:
                rec.append({"method": nm, "k": 0, "study": g[j], "cover": abs(y[j] - pt[i]) <= q[nm], "width": 2 * q[nm]})
            qq = qm[sub[j]]
            rec.append({"method": "mondrian_subtype", "k": 0, "study": g[j], "cover": abs(y[j] - pt[i]) <= qq, "width": 2 * qq})
        # few-shot adapted
        for k in [1, 3, 5]:
            qk = qhat(adapted_residuals(y[tr], p_in_g, g[tr], k))
            for s in np.unique(g[te]):
                idx = np.where(g[te] == s)[0]
                if len(idx) <= k + 1:
                    continue
                for _ in range(10):
                    anc = RNG.choice(idx, k, replace=False); qi = np.setdiff1d(idx, anc)
                    pa = pt[qi] + np.mean(y[te][anc] - pt[anc])
                    for a, b in zip(y[te][qi], pa):
                        rec.append({"method": f"study_fewshot", "k": k, "study": s, "cover": abs(a - b) <= qk, "width": 2 * qk})
    R = pd.DataFrame(rec)
    out = []
    for (mth, k), z in R.groupby(["method", "k"]):
        sc = z.groupby("study").cover.mean()
        out.append({"target": tgt, "method": mth, "k": k, "coverage": z.cover.mean(), "mean_width": z.width.mean(),
                    "study_cov_median": sc.median(), "frac_studies_cov_below_0.8": float((sc < 0.8).mean())})
    O = pd.DataFrame(out); print(O.to_string(), flush=True)
    return O


if __name__ == "__main__":
    tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    pd.concat([run(t) for t in tg]).to_csv(RES / "06_study_conformal.csv", index=False)
