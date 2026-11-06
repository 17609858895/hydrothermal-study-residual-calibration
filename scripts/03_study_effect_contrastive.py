"""Core experiment: does pooled ML learn chemistry or the lab?

Outer loop: 5-fold GroupKFold by source study (test studies unseen).
Models (same inputs, same LightGBM budget):
  P   pooled LightGBM on absolute yields
  RE  random-effect (BLUP-shrunk study intercept) backfitting LightGBM  ~ MERF/GPBoost-type closest method
  SC  Study-Contrastive learner: LightGBM trained only on WITHIN-study pairs, target y_i - y_j,
      features [x_i, x_j, x_i - x_j], antisymmetrised at inference. Study-level bias cancels by design.
Evaluation on held-out studies:
  zero-shot absolute R2 (P, RE);
  within-study contrast R2 / sign accuracy on all test pairs (|dy|>=2);
  few-shot adaptation with k anchor runs from the new study (k=1,3,5; 30 random draws).
Also: ICC of out-of-fold residuals (between-study share of unexplained variance).
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

TARGETS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
RNG = np.random.default_rng(SEED)
LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED, n_jobs=6)
KAPPA = 3.0          # BLUP shrinkage: u_s = n_s/(n_s+kappa) * mean residual
MAX_PAIRS = 400      # ordered within-study pairs sampled per training study
K_SHOTS = [0, 1, 3, 5]
N_DRAW = 30


def icc(resid, groups):
    d = pd.DataFrame({"r": resid, "g": groups})
    try:
        m = smf.mixedlm("r ~ 1", d, groups=d["g"]).fit(reml=True)
        su = float(m.cov_re.iloc[0, 0]); se = float(m.scale)
        return su / (su + se)
    except Exception:
        return np.nan


def fit_re(X, y, g, n_iter=4):
    u = pd.Series(0.0, index=np.unique(g))
    m = None
    for _ in range(n_iter):
        m = lgb.LGBMRegressor(**LGB).fit(X, y - u.loc[g].values)
        r = pd.Series(y - m.predict(X)).groupby(g).agg(["mean", "size"])
        u = r["mean"] * r["size"] / (r["size"] + KAPPA)
    return m, u


def make_pairs(X, y, g, rng, max_pairs=MAX_PAIRS):
    I, J = [], []
    for s in np.unique(g):
        idx = np.where(g == s)[0]
        if len(idx) < 2:
            continue
        a = rng.choice(idx, size=min(max_pairs, len(idx) * (len(idx) - 1)), replace=True)
        b = rng.choice(idx, size=len(a), replace=True)
        keep = a != b
        I.append(a[keep]); J.append(b[keep])
    I = np.concatenate(I); J = np.concatenate(J)
    return I, J


def pair_feat(Xa, Xb):
    return np.hstack([Xa, Xb, Xa - Xb])


def sc_predict(m, Xa, Xb):
    """antisymmetrised contrast prediction g(a,b) = y_a - y_b"""
    return 0.5 * (m.predict(pair_feat(Xa, Xb)) - m.predict(pair_feat(Xb, Xa)))


def run(tgt):
    df = load()
    d = df[df[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values; g = d["study_id"].values
    folds = splits(d, "group")
    oof_P = np.zeros(len(d)); oof_RE = np.zeros(len(d))
    contrast_rows, shot_rows = [], []
    for fi, (tr, te) in enumerate(folds):
        t0 = time.time()
        mP = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr])
        mRE, _ = fit_re(X[tr], y[tr], g[tr])
        I, J = make_pairs(X[tr], y[tr], g[tr], RNG)
        mSC = lgb.LGBMRegressor(**{**LGB, "n_estimators": 800}).fit(pair_feat(X[tr][I], X[tr][J]), y[tr][I] - y[tr][J])
        oof_P[te] = mP.predict(X[te]); oof_RE[te] = mRE.predict(X[te])
        # ---- within-study contrasts on test studies
        for s in np.unique(g[te]):
            idx = te[g[te] == s]
            if len(idx) < 3:
                continue
            n_s = len(idx)
            A, B = np.meshgrid(np.arange(n_s), np.arange(n_s), indexing="ij")
            Gm = sc_predict(mSC, X[idx][A.ravel()], X[idx][B.ravel()]).reshape(n_s, n_s)  # Gm[a,b] ~ y_a - y_b
            a, b = np.triu_indices(n_s, 1)
            ia, ib = idx[a], idx[b]
            dy = y[ia] - y[ib]
            pP = oof_P[ia] - oof_P[ib]; pRE = oof_RE[ia] - oof_RE[ib]; pSC = Gm[a, b]
            contrast_rows.append(pd.DataFrame({"study": s, "dy": dy, "P": pP, "RE": pRE, "SC": pSC}))
            pos = {v: i for i, v in enumerate(idx)}
            # ---- few-shot anchors
            for k in K_SHOTS:
                if n_s <= k + 1:
                    continue
                for r in range(N_DRAW if k > 0 else 1):
                    anc = RNG.choice(idx, size=k, replace=False) if k > 0 else np.array([], int)
                    qry = np.setdiff1d(idx, anc)
                    out = {"P": oof_P[qry].copy(), "RE": oof_RE[qry].copy()}
                    if k > 0:
                        out["P"] += np.mean(y[anc] - oof_P[anc])
                        off = np.sum(y[anc] - oof_RE[anc]) / (k + KAPPA)        # BLUP offset
                        out["RE"] += off
                        qp = np.array([pos[v] for v in qry]); ap = np.array([pos[v] for v in anc])
                        sc = np.mean(y[anc][None, :] + Gm[np.ix_(qp, ap)], axis=1)
                        out["SC"] = sc
                        out["SC+RE"] = 0.5 * (sc + out["RE"])
                    for mname, p in out.items():
                        shot_rows.append({"study": s, "k": k, "draw": r, "model": mname,
                                          "sse": float(np.sum((y[qry] - p) ** 2)), "n": len(qry),
                                          "rmse_study": float(np.sqrt(np.mean((y[qry] - p) ** 2)))})
        print(f"{tgt} fold {fi} done {time.time()-t0:.0f}s pairs={len(I)}", flush=True)
    C = pd.concat(contrast_rows)
    C = C[np.abs(C.dy) >= 2.0]
    S = pd.DataFrame(shot_rows)
    summ = {"target": tgt, "n": len(d), "studies": int(d.study_id.nunique())}
    for nm, p in [("P", oof_P), ("RE", oof_RE)]:
        summ[f"zeroshot_{nm}"] = metrics(y, p)
        summ[f"ICC_resid_{nm}"] = icc(y - p, g)
    summ["ICC_raw_y"] = icc(y - y.mean(), g)
    for nm in ["P", "RE", "SC"]:
        r2 = 1 - np.sum((C.dy - C[nm]) ** 2) / np.sum((C.dy - C.dy.mean()) ** 2)
        acc = float(np.mean(np.sign(C.dy) == np.sign(C[nm])))
        # study-median sign accuracy
        sa = C.groupby("study").apply(lambda z: np.mean(np.sign(z.dy) == np.sign(z[nm]))).median()
        summ[f"contrast_{nm}"] = {"R2_delta": float(r2), "sign_acc": acc, "sign_acc_study_median": float(sa),
                                  "n_pairs": int(len(C)), "n_studies": int(C.study.nunique())}
    agg = S.groupby(["k", "model"]).apply(lambda z: pd.Series({
        "RMSE_pooled": np.sqrt(z.sse.sum() / z.n.sum()), "RMSE_study_median": z.rmse_study.median()})).reset_index()
    summ["fewshot"] = agg.to_dict("records")
    C.to_csv(RES / f"03_contrasts_{tgt}.csv", index=False)
    S.to_csv(RES / f"03_fewshot_raw_{tgt}.csv", index=False)
    pd.DataFrame({"study_id": g, "y": y, "P": oof_P, "RE": oof_RE}).to_csv(RES / f"03_oof_{tgt}.csv", index=False)
    return summ


if __name__ == "__main__":
    allres = {}
    for t in TARGETS:
        allres[t] = run(t)
        print(json.dumps(allres[t], indent=1, default=float), flush=True)
    with open(RES / f"03_summary_{'_'.join(TARGETS)}.json", "w") as f:
        json.dump(allres, f, indent=1, default=float)
