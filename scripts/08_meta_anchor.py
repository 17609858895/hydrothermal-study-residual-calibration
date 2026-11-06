"""Episodic meta-learned anchor calibration (MAC) for a new laboratory.

Setting: a new study supplies k anchor runs (measured yields). Predict its remaining runs.
Base: pooled LightGBM f(x). Residual r = y - f(x) is strongly study-correlated (ICC ~0.6-0.7)
but NOT constant inside a study; it decays with condition dissimilarity.

MAC learns r_q = h(phi(q, anchors)) from simulated episodes over training studies, using only
study-held-out (inner OOF) base predictions so the residuals it learns from are honest.
phi: base prediction, mean anchor residual, similarity-weighted anchor residual, nearest-anchor
residual and distance, same-feedstock-anchor residual, anchor y spread, k, and query conditions.

Comparators (same base predictions, same anchors):
  offset   constant mean residual (local recalibration)
  BLUP     shrunk offset sum(r_a)/(k+kappa)  (random-intercept)
  kernel   similarity-weighted residual, no learning
  MAC      learned calibrator
  MAC-noSim  ablation: learned calibrator with only k, base pred, mean residual (no similarity terms)
Episode-conformal: MAC is cross-fitted over two study halves; absolute residuals of the half not
used for fitting give the conformal quantile, so coverage refers to adapted predictions.
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

RNG = np.random.default_rng(SEED)
LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED, n_jobs=8)
META = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=SEED, n_jobs=8)
KS = [1, 3, 5]; KAPPA = 3.0; ALPHA = 0.10
COND = ["T", "log_t", "SL", "CatR", "SBR"]
MATCH = ["Feedstock", "catalyst_class", "solvent_class"]


def oof(X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]).predict(X[te])
    return p


def dist_matrix(C, M, qi, ai):
    """standardised condition distance + 1 per categorical mismatch"""
    dq = C[qi][:, None, :] - C[ai][None, :, :]
    dn = np.sqrt(np.nansum(dq ** 2, axis=2) / C.shape[1])
    mm = (M[qi][:, None, :] != M[ai][None, :, :]).sum(axis=2)
    return dn + mm


def episode_features(y, p, C, M, Xq_cond, qi, ai):
    r_a = y[ai] - p[ai]
    D = dist_matrix(C, M, qi, ai)
    w = np.exp(-D / 0.5); w = w / (w.sum(axis=1, keepdims=True) + 1e-9)
    nn = D.argmin(axis=1)
    same_fs = (M[qi][:, None, 0] == M[ai][None, :, 0])
    sf_res = np.where(same_fs.any(axis=1), (same_fs * r_a[None, :]).sum(1) / np.maximum(same_fs.sum(1), 1), np.nan)
    k = len(ai)
    F = np.column_stack([
        np.full(len(qi), k), p[qi], np.full(len(qi), r_a.mean()), (w * r_a[None, :]).sum(1),
        r_a[nn], D.min(axis=1), D.mean(axis=1), sf_res, np.full(len(qi), y[ai].std() if k > 1 else 0.0),
        np.full(len(qi), r_a.std() if k > 1 else 0.0), p[qi] - p[ai].mean(), Xq_cond[qi]])
    kern = (w * r_a[None, :]).sum(1)
    return F, r_a, kern


def make_episodes(y, p, C, M, Xc, g, studies, n_draw=8):
    Fs, T, meta = [], [], []
    for s in studies:
        idx = np.where(g == s)[0]
        for k in KS:
            if len(idx) <= k + 1:
                continue
            for _ in range(n_draw):
                ai = RNG.choice(idx, k, replace=False); qi = np.setdiff1d(idx, ai)
                F, r_a, kern = episode_features(y, p, C, M, Xc, qi, ai)
                Fs.append(F); T.append(y[qi] - p[qi])
                meta.append(pd.DataFrame({"study": s, "k": k, "q": qi, "offset": r_a.mean(),
                                          "blup": r_a.sum() / (k + KAPPA), "kernel": kern}))
    return np.vstack(Fs), np.concatenate(T), pd.concat(meta, ignore_index=True)


NOSIM = [0, 1, 2]  # columns: k, base pred, mean anchor residual


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values; g = d.study_id.values
    C = d[COND].astype(float).values
    C = (C - np.nanmean(C, 0)) / (np.nanstd(C, 0) + 1e-9)        # label-free scaling of inputs
    M = d[MATCH].astype(str).values
    Xc = d[COND].astype(float).values
    rows = []
    for fo, (tr, te) in enumerate(splits(d, "group")):
        t0 = time.time()
        dtr = d.iloc[tr].reset_index(drop=True)
        p_in = np.zeros(len(d)); p_in[tr] = oof(X[tr], y[tr], splits(dtr, "group", seed=SEED + 3))
        base = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]); p_te = np.zeros(len(d)); p_te[te] = base.predict(X[te])
        tr_st = np.unique(g[tr]); RNG.shuffle(tr_st); half = [tr_st[::2], tr_st[1::2]]
        F_all, T_all, meta_all, models, models_ns, calres = [], [], [], [], [], []
        epi = [make_episodes(y, p_in, C, M, Xc, g, h) for h in half]
        for h in range(2):
            Fa, Ta, _ = epi[h]; Fb, Tb, mb = epi[1 - h]
            m = lgb.LGBMRegressor(**META).fit(Fa, Ta); mns = lgb.LGBMRegressor(**META).fit(Fa[:, NOSIM], Ta)
            models.append(m); models_ns.append(mns)
            calres.append(pd.DataFrame({"k": mb.k.values, "abs": np.abs(Tb - m.predict(Fb)),
                                        "abs_off": np.abs(Tb - mb.offset.values)}))
        cal = pd.concat(calres)
        qk = {k: np.quantile(z["abs"], min(1, np.ceil((len(z) + 1) * (1 - ALPHA)) / len(z))) for k, z in cal.groupby("k")}
        qk_off = {k: np.quantile(z["abs_off"], min(1, np.ceil((len(z) + 1) * (1 - ALPHA)) / len(z))) for k, z in cal.groupby("k")}
        # ---- test episodes
        for s in np.unique(g[te]):
            idx = np.where(g == s)[0]
            for k in KS:
                if len(idx) <= k + 1:
                    continue
                for dr in range(20):
                    ai = RNG.choice(idx, k, replace=False); qi = np.setdiff1d(idx, ai)
                    F, r_a, kern = episode_features(y, p_te, C, M, Xc, qi, ai)
                    pr = {"base(k=0)": p_te[qi], "offset": p_te[qi] + r_a.mean(),
                          "BLUP": p_te[qi] + r_a.sum() / (k + KAPPA), "kernel": p_te[qi] + kern,
                          "MAC": p_te[qi] + np.mean([m.predict(F) for m in models], axis=0),
                          "MAC-noSim": p_te[qi] + np.mean([m.predict(F[:, NOSIM]) for m in models_ns], axis=0)}
                    for nm, pp in pr.items():
                        e = y[qi] - pp
                        rec = {"fold": fo, "study": s, "k": k, "draw": dr, "model": nm, "sse": float(np.sum(e ** 2)),
                               "n": len(qi), "sst_q": float(np.sum((y[qi] - y[tr].mean()) ** 2)),
                               "rmse_study": float(np.sqrt(np.mean(e ** 2)))}
                        if nm == "MAC":
                            rec.update(cover=float(np.mean(np.abs(e) <= qk[k])), width=2 * qk[k])
                        if nm == "offset":
                            rec.update(cover=float(np.mean(np.abs(e) <= qk_off[k])), width=2 * qk_off[k])
                        rows.append(rec)
        print(f"{tgt} fold {fo} {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RES / f"08_meta_anchor_raw_{tgt}.csv", index=False)
    S = R.groupby(["k", "model"]).apply(lambda z: pd.Series({
        "RMSE": np.sqrt(z.sse.sum() / z.n.sum()), "R2_vs_trainmean": 1 - z.sse.sum() / z.sst_q.sum(),
        "RMSE_study_median": z.rmse_study.median(),
        "coverage": np.average(z.cover, weights=z.n) if "cover" in z and z.cover.notna().any() else np.nan,
        "width": z.width.mean() if "width" in z else np.nan})).reset_index()
    S["target"] = tgt
    print(S.to_string(), flush=True)
    return S


if __name__ == "__main__":
    tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    pd.concat([run(t) for t in tg]).to_csv(RES / "08_meta_anchor_summary.csv", index=False)
