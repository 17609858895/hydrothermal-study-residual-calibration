"""Diagnostic: how much WITHIN-study variation is explainable, and is it transferable?

A) demeaned target y~ = y - mean_study(y). LightGBM on [X~, X] (X~ demeaned features):
   random-row CV (other runs of the same study seen)  vs  study-grouped CV (new study).
B) one-factor-at-a-time (OFAT) contrasts inside held-out studies: pairs with identical feedstock
   string and identical categorical conditions that differ in exactly one of T / t / SL / CatR.
   Compare sign accuracy and R2 of the predicted contrast for pooled (P) vs within-study (FE) model.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, random_state=SEED, n_jobs=8)
FACT = {"T": 5.0, "t": 1.0, "SL": 0.5, "CatR": 0.005}


def demean(A, g):
    A = pd.DataFrame(A); return (A - A.groupby(g).transform("mean")).values


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values; g = d.study_id.values
    Xt = np.hstack([demean(X, g), X]); yt = demean(y.reshape(-1, 1), g).ravel()
    out = {"target": tgt, "within_var_share": float(np.var(yt) / np.var(y))}
    pred = {}
    for kind in ["random", "group"]:
        p = np.zeros(len(y)); pp = np.zeros(len(y))
        for tr, te in splits(d, kind):
            p[te] = lgb.LGBMRegressor(**LGB).fit(Xt[tr], yt[tr]).predict(Xt[te])
            pp[te] = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]).predict(X[te])
        # within-study R2 of demeaned prediction (re-demean predictions within study)
        pd_ = demean(p.reshape(-1, 1), g).ravel(); ppd = demean(pp.reshape(-1, 1), g).ravel()
        out[f"FE_withinR2_{kind}"] = float(1 - np.sum((yt - pd_) ** 2) / np.sum(yt ** 2))
        out[f"P_withinR2_{kind}"] = float(1 - np.sum((yt - ppd) ** 2) / np.sum(yt ** 2))
        pred[kind] = (pd_, ppd)
    # OFAT pairs on group-CV predictions
    fe, pl = pred["group"]
    cats = CAT + ["Feedstock"]
    key = d[cats].astype(str).agg("|".join, axis=1).values
    rows = []
    for s in np.unique(g):
        idx = np.where(g == s)[0]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if key[i] != key[j]:
                    continue
                diff = [f for f, tol in FACT.items()
                        if not (np.isnan(d[f].iat[i]) and np.isnan(d[f].iat[j])) and abs(np.nan_to_num(d[f].iat[i]) - np.nan_to_num(d[f].iat[j])) > tol]
                if len(diff) != 1:
                    continue
                rows.append({"study": s, "factor": diff[0], "dy": y[i] - y[j], "FE": fe[i] - fe[j], "P": pl[i] - pl[j]})
    R = pd.DataFrame(rows)
    R = R[np.abs(R.dy) >= 1.0]
    res = {}
    for f, z in list(R.groupby("factor")) + [("ALL", R)]:
        res[f] = {"n_pairs": len(z), "n_studies": int(z.study.nunique())}
        for m in ["FE", "P"]:
            res[f][f"{m}_sign_acc"] = float(np.mean(np.sign(z.dy) == np.sign(z[m])))
            res[f][f"{m}_R2"] = float(1 - np.sum((z.dy - z[m]) ** 2) / np.sum((z.dy - z.dy.mean()) ** 2))
            res[f][f"{m}_corr"] = float(np.corrcoef(z.dy, z[m])[0, 1])
    out["OFAT"] = res
    R.to_csv(RES / f"07_ofat_pairs_{tgt}.csv", index=False)
    return out


if __name__ == "__main__":
    tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    allr = {t: run(t) for t in tg}
    print(json.dumps(allr, indent=1))
    json.dump(allr, open(RES / "07_within_diag.json", "w"), indent=1)
