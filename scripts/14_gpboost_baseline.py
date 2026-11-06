"""Nearest-neighbour baseline for SR-GP: GPBoost (tree boosting + grouped random effects + GP).

GPBoost (Sigrist, JMLR 2022) learns the trees F(x) and the random-effect covariance jointly on the
training studies. A trained GPBoost model is applied to a new study the same way SR-GP is: trees and
covariance parameters stay fixed, the k anchor runs give the posterior random effect of the new study
(computed with GPBoost's own GPModel.predict(y=anchor residuals, cov_pars=learned)), and the query runs
are predicted as F(x_q) + E[b | anchors].

Variants
  GPB-RE      : trees + random study intercept                      (~ mixed-effects boosting / MERF)
  GPB-RE+GP   : trees + random study intercept + per-study GP over standardised process conditions
                (gp_coords = COND, cluster_ids = study, shared hyper-parameters, Gaussian kernel)
                = SR-GP without the categorical-mismatch decay, but with trees fitted jointly.
  SR-GP@GPB   : SR-GP residual kernel fitted on GPB-RE fixed-effect residuals (inner OOF), i.e. the
                SR-GP correction on top of jointly de-confounded trees.
Compared on the SAME anchor draws with: pooled LightGBM (k=0), constant offset, SR-GP (09_srgp.py).
R2 as in 09: 1 - SSE/SST on query runs, SST about the outer-training mean.
Boosting: learning_rate 0.1, 1500 rounds (selected by study-grouped inner CV on fold 0; early stopping
reached the cap for both targets). GPBoost does not allow bagging.
"""
import sys, importlib.util, warnings, time, os
from pathlib import Path
import numpy as np, pandas as pd
import gpboost as gpb
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import *
spec = importlib.util.spec_from_file_location("srgp", HERE / "09_srgp.py"); S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

RNG = np.random.default_rng(SEED + 14)
KS = [1, 3, 5]
NDRAW = int(os.environ.get("GPB_DRAWS", 20))
GPB = dict(learning_rate=0.1, num_leaves=31, min_data_in_leaf=10, feature_fraction=0.8,
           lambda_l2=1.0, verbose=-1, num_threads=8, seed=SEED)
NROUND = int(os.environ.get("GPB_ROUNDS", 1500))
VARS = ["GPB-RE", "GPB-RE+GP"]


def fit_gpb(X, y, grp, coords, variant):
    if variant == "GPB-RE":
        gm = gpb.GPModel(group_data=grp, likelihood="gaussian")
    else:
        gm = gpb.GPModel(group_data=grp, gp_coords=coords, cov_function="gaussian", cluster_ids=grp,
                         likelihood="gaussian")
    bst = gpb.train(params=GPB, train_set=gpb.Dataset(X, y), gp_model=gm, num_boost_round=NROUND)
    return bst, np.asarray(gm.get_cov_pars()).ravel()


def fixed_effect(bst, variant, X, grp, coords):
    kw = dict(data=X, group_data_pred=grp, predict_var=False, pred_latent=True)
    if variant != "GPB-RE":
        kw.update(gp_coords_pred=coords, cluster_ids_pred=grp)
    return np.asarray(bst.predict(**kw)["fixed_effect"])


def re_posterior(variant, cp, ra, Ca, Cq):
    """Posterior mean of the new study's random effect at the query runs (GPBoost's own predictor)."""
    k, nq = len(ra), len(Cq)
    if variant == "GPB-RE":
        gm = gpb.GPModel(group_data=np.zeros(k), likelihood="gaussian")
        out = gm.predict(y=ra, cov_pars=cp, group_data_pred=np.zeros(nq), predict_response=False)
    else:
        gm = gpb.GPModel(group_data=np.zeros(k), gp_coords=Ca, cov_function="gaussian", likelihood="gaussian")
        out = gm.predict(y=ra, cov_pars=cp, group_data_pred=np.zeros(nq), gp_coords_pred=Cq, predict_response=False)
    return np.asarray(out["mu"]).ravel()


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values.astype(float); y = d[tgt].values; g = d.study_id.values
    gi = pd.factorize(g)[0].astype(int)
    Craw = d[S.COND].astype(float).values; M = d[S.MATCH].astype(str).values
    rows, covp = [], []
    for fo, (tr, te) in enumerate(splits(d, "group")):
        t0 = time.time()
        mu_c = np.nanmean(Craw[tr], 0); sd_c = np.nanstd(Craw[tr], 0) + 1e-9
        C = np.nan_to_num((Craw - mu_c) / sd_c)
        dtr = d.iloc[tr].reset_index(drop=True)
        inner = splits(dtr, "group", seed=SEED + 3)
        # SR-GP on pooled LightGBM (as in 09)
        r_in = np.zeros(len(d)); r_in[tr] = y[tr] - S.oof(X[tr], y[tr], inner)
        p_te = np.zeros(len(d)); p_te[te] = S.make_base().fit(X[tr], y[tr]).predict(X[te])
        th = S.fit_theta(C, M, r_in, g, np.unique(g[tr]), "full")
        # GPBoost variants trained on the training studies
        F, CP = {}, {}
        for v in VARS:
            tv = time.time()
            bst, cp = fit_gpb(X[tr], y[tr], gi[tr], C[tr], v)
            F[v] = np.zeros(len(d)); F[v][te] = fixed_effect(bst, v, X[te], gi[te], C[te]); CP[v] = cp
            covp.append({"fold": fo, "variant": v, "sec": time.time() - tv, **{f"cp{i}": float(c) for i, c in enumerate(cp)}})
        # SR-GP kernel on top of GPB-RE trees: inner OOF fixed-effect residuals on the training studies
        r_g = np.zeros(len(d))
        for itr, ite in inner:
            a, b = tr[itr], tr[ite]
            bst, _ = fit_gpb(X[a], y[a], gi[a], C[a], "GPB-RE")
            r_g[b] = y[b] - fixed_effect(bst, "GPB-RE", X[b], gi[b], C[b])
        th_g = S.fit_theta(C, M, r_g, g, np.unique(g[tr]), "full")
        covp.append({"fold": fo, "variant": "theta_SRGP", **{f"cp{i}": float(c) for i, c in enumerate(th)}})
        covp.append({"fold": fo, "variant": "theta_SRGP@GPB", **{f"cp{i}": float(c) for i, c in enumerate(th_g)}})
        for s in np.unique(g[te]):
            idx = np.where(g == s)[0]
            for k in KS:
                if len(idx) <= k + 1: continue
                for dr in range(NDRAW):
                    ai = RNG.choice(idx, k, replace=False); qi = np.setdiff1d(idx, ai)
                    ra = y[ai] - p_te[ai]
                    mu, _ = S.posterior(C[ai], M[ai], ra, C[qi], M[qi], th, "full")
                    preds = {"pooled (k=0)": p_te[qi], "offset": p_te[qi] + ra.mean(), "SR-GP": p_te[qi] + mu}
                    for v in VARS:
                        preds[f"{v} (k=0)"] = F[v][qi]
                        preds[v] = F[v][qi] + re_posterior(v, CP[v], y[ai] - F[v][ai], C[ai], C[qi])
                    rg = y[ai] - F["GPB-RE"][ai]
                    mu_g, _ = S.posterior(C[ai], M[ai], rg, C[qi], M[qi], th_g, "full")
                    preds["SR-GP@GPB"] = F["GPB-RE"][qi] + mu_g
                    for nm, p in preds.items():
                        e = y[qi] - p
                        rows.append({"fold": fo, "study": s, "k": k, "draw": dr, "model": nm, "sse": float(np.sum(e ** 2)),
                                     "n": len(qi), "sst": float(np.sum((y[qi] - y[tr].mean()) ** 2)),
                                     "rmse_study": float(np.sqrt(np.mean(e ** 2)))})
        print(f"{tgt} fold {fo} {time.time()-t0:.0f}s cov GPB-RE={np.round(CP['GPB-RE'],1)} GPB-RE+GP={np.round(CP['GPB-RE+GP'],2)}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(RES / f"14_gpboost_raw_{tgt}.csv", index=False)
    pd.DataFrame(covp).to_csv(RES / f"14_gpboost_covpars_{tgt}.csv", index=False)
    Sm = R.groupby(["k", "model"]).apply(lambda z: pd.Series({
        "R2": 1 - z.sse.sum() / z.sst.sum(), "RMSE": np.sqrt(z.sse.sum() / z.n.sum()),
        "RMSE_study_median": z.rmse_study.median()})).reset_index()
    # paired per-episode comparison: share of (study, draw) episodes where SR-GP has the lower RMSE
    P = R.pivot_table(index=["fold", "study", "k", "draw"], columns="model", values="rmse_study").reset_index()
    for b in ["offset", "GPB-RE", "GPB-RE+GP", "SR-GP@GPB"]:
        w = P.groupby("k").apply(lambda z: float(np.mean(z["SR-GP"] < z[b])))
        Sm.loc[Sm.model == b, "frac_episodes_SRGP_better"] = Sm.loc[Sm.model == b, "k"].map(w).values
    Sm["target"] = tgt
    print(Sm.to_string(), flush=True)
    return Sm


if __name__ == "__main__":
    tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    pd.concat([run(t) for t in tg]).to_csv(RES / f"14_gpboost_summary_{'_'.join(tg)}.csv", index=False)
