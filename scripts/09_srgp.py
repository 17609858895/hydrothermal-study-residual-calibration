"""Study-Residual Gaussian Process (SR-GP) for few-shot transfer to a new laboratory.

Base learner f(x): pooled LightGBM (any learner works). Residual r = y - f(x).
Within a study s, residuals are modelled as
    r_i = u_s + v_s(c_i) + e_i,
    cov(r_i, r_j | same study) = s_u^2 + s_v^2 * exp(-||c_i-c_j||^2 / (2 l^2)) * rho^{m_ij} + s_e^2 [i=j]
    cov(r_i, r_j | different studies) = 0
c = standardised process conditions (T, log t, solid loading, catalyst ratio, solvent ratio);
m_ij = number of mismatches in (feedstock, catalyst class, solvent class).
Hyper-parameters theta = (s_u, s_v, l, rho, s_e) are learned by maximising the summed Gaussian
log-marginal likelihood of study-held-out (inner OOF) residuals over training studies.
A new study with k anchor runs gets the GP posterior mean/variance for its remaining runs.
Special cases (ablations): s_v=0 -> random-intercept BLUP; s_u=0 -> purely local kernel.
Normalised conformal: score |y - yhat| / sd_post, quantile from episodes on the other half of the
training studies (hyper-parameters fitted on one half, calibrated on the other, then swapped).
Also writes the empirical within-study residual semivariogram vs condition distance.
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

RNG = np.random.default_rng(SEED)
LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED, n_jobs=8)
import os
BASE = os.environ.get("SRGP_BASE", "lightgbm").lower()      # lightgbm | catboost
SUF = "" if BASE == "lightgbm" else f"_{BASE}"


def make_base():
    if BASE == "catboost":
        import catboost as cb
        return cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0,
                                    thread_count=8, random_seed=SEED)
    return lgb.LGBMRegressor(**LGB)


KS = [1, 3, 5]; ALPHA = 0.10; KAPPA = 3.0
COND = ["T", "log_t", "SL", "CatR", "SBR"]
MATCH = ["Feedstock", "catalyst_class", "solvent_class"]


def oof(X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = make_base().fit(X[tr], y[tr]).predict(X[te])
    return p


def kern(Ci, Cj, Mi, Mj, th, mode="full"):
    su, sv, l, rho, se = th
    d2 = ((Ci[:, None, :] - Cj[None, :, :]) ** 2).sum(-1)
    mm = (Mi[:, None, :] != Mj[None, :, :]).sum(-1)
    K = np.zeros_like(d2)
    if mode in ("full", "intercept"):
        K += su ** 2
    if mode in ("full", "local"):
        K += sv ** 2 * np.exp(-d2 / (2 * l ** 2)) * rho ** mm
    return K


def unpack(z, mode):
    su, sv, l, se = np.exp(z[:4]); rho = 1 / (1 + np.exp(-z[4]))
    if mode == "intercept": sv = 0.0
    if mode == "local": su = 0.0
    return su, sv, l, rho, se


def nlml(z, groups, mode):
    th = unpack(z, mode); tot = 0.0
    for C, M, r in groups:
        K = kern(C, C, M, M, th, mode) + (th[4] ** 2 + 1e-6) * np.eye(len(r))
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        a = cho_solve(cf, r)
        tot += 0.5 * r @ a + np.sum(np.log(np.diag(cf[0]))) + 0.5 * len(r) * np.log(2 * np.pi)
    return tot


def fit_theta(C, M, r, g, studies, mode):
    groups = []
    for s in studies:
        idx = np.where(g == s)[0]
        if len(idx) >= 2:
            if len(idx) > 120:                       # cap very large studies for speed
                idx = RNG.choice(idx, 120, replace=False)
            groups.append((C[idx], M[idx], r[idx]))
    sd = np.std(r)
    z0 = np.array([np.log(sd * 0.7), np.log(sd * 0.5), np.log(1.0), np.log(sd * 0.4), 0.0])
    res = minimize(nlml, z0, args=(groups, mode), method="L-BFGS-B")
    return unpack(res.x, mode)


def posterior(Ca, Ma, ra, Cq, Mq, th, mode):
    Kaa = kern(Ca, Ca, Ma, Ma, th, mode) + (th[4] ** 2 + 1e-6) * np.eye(len(ra))
    Kqa = kern(Cq, Ca, Mq, Ma, th, mode)
    cf = cho_factor(Kaa, lower=True)
    mu = Kqa @ cho_solve(cf, ra)
    prior = (th[0] ** 2 if mode != "local" else 0) + (th[1] ** 2 if mode != "intercept" else 0)
    var = prior - np.einsum("ij,ji->i", Kqa, cho_solve(cf, Kqa.T)) + th[4] ** 2
    return mu, np.sqrt(np.maximum(var, 1e-6))


def variogram(C, M, r, g, nb=12):
    rows = []
    for s in np.unique(g):
        idx = np.where(g == s)[0]
        if len(idx) < 3: continue
        a, b = np.triu_indices(len(idx), 1)
        ia, ib = idx[a], idx[b]
        dist = np.sqrt(((C[ia] - C[ib]) ** 2).sum(1)); mm = (M[ia] != M[ib]).sum(1)
        rows.append(pd.DataFrame({"dist": dist, "mismatch": mm, "sv": 0.5 * (r[ia] - r[ib]) ** 2}))
    W = pd.concat(rows)
    # between-study semivariance (sill) from random cross-study pairs
    i = RNG.integers(0, len(r), 200000); j = RNG.integers(0, len(r), 200000); keep = g[i] != g[j]
    sill = float(np.mean(0.5 * (r[i[keep]] - r[j[keep]]) ** 2))
    W["bin"] = pd.qcut(W.dist, nb, duplicates="drop")
    V = W.groupby(["bin", (W.mismatch > 0).rename("any_mismatch")]).agg(dist=("dist", "median"), gamma=("sv", "mean"), n=("sv", "size")).reset_index()
    V["sill_between"] = sill
    return V.drop(columns="bin")


def qhat(x):
    n = len(x); return np.quantile(x, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n))


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values; g = d.study_id.values
    Craw = d[COND].astype(float).values
    M = d[MATCH].astype(str).values
    rows, thetas, vgs = [], [], []
    for fo, (tr, te) in enumerate(splits(d, "group")):
        t0 = time.time()
        mu_c = np.nanmean(Craw[tr], 0); sd_c = np.nanstd(Craw[tr], 0) + 1e-9
        C = np.nan_to_num((Craw - mu_c) / sd_c)
        dtr = d.iloc[tr].reset_index(drop=True)
        r_in = np.zeros(len(d)); r_in[tr] = y[tr] - oof(X[tr], y[tr], splits(dtr, "group", seed=SEED + 3))
        p_te = np.zeros(len(d)); p_te[te] = make_base().fit(X[tr], y[tr]).predict(X[te])
        tr_st = np.unique(g[tr])
        if fo == 0:
            vgs.append(variogram(C[tr], M[tr], r_in[tr], g[tr]))
        th = {m: fit_theta(C, M, r_in, g, tr_st, m) for m in ["full", "intercept", "local"]}
        thetas.append({"fold": fo, **{f"{m}_{nm}": float(v) for m in th for nm, v in zip(["su", "sv", "l", "rho", "se"], th[m])}})
        # conformal calibration: theta from half A, scores on half B (and swap)
        perm = RNG.permutation(tr_st); halves = [perm[::2], perm[1::2]]
        scores = {k: [] for k in KS}; scores_off = {k: [] for k in KS}
        for h in range(2):
            thA = fit_theta(C, M, r_in, g, halves[h], "full")
            for s in halves[1 - h]:
                idx = np.where(g == s)[0]
                for k in KS:
                    if len(idx) <= k + 1: continue
                    for _ in range(5):
                        ai = RNG.choice(idx, k, replace=False); qi = np.setdiff1d(idx, ai)
                        mu, sd = posterior(C[ai], M[ai], r_in[ai], C[qi], M[qi], thA, "full")
                        scores[k].append(np.abs(r_in[qi] - mu) / sd)
                        scores_off[k].append(np.abs(r_in[qi] - r_in[ai].mean()))
        qn = {k: qhat(np.concatenate(v)) for k, v in scores.items()}
        qo = {k: qhat(np.concatenate(v)) for k, v in scores_off.items()}
        for s in np.unique(g[te]):
            idx = np.where(g == s)[0]
            for k in KS:
                if len(idx) <= k + 1: continue
                for dr in range(20):
                    ai = RNG.choice(idx, k, replace=False); qi = np.setdiff1d(idx, ai)
                    ra = y[ai] - p_te[ai]
                    D = np.sqrt(((C[qi][:, None] - C[ai][None]) ** 2).sum(-1) / C.shape[1]) + (M[qi][:, None] != M[ai][None]).sum(-1)
                    w = np.exp(-D / 0.5); w /= w.sum(1, keepdims=True)
                    preds = {"offset": (ra.mean(), None), "BLUP(fixed)": (ra.sum() / (k + KAPPA), None),
                             "kernel(fixed)": ((w * ra[None]).sum(1), None)}
                    for m in ["full", "intercept", "local"]:
                        mu, sd = posterior(C[ai], M[ai], ra, C[qi], M[qi], th[m], m)
                        preds[{"full": "SR-GP", "intercept": "SR-GP(no local)", "local": "SR-GP(no intercept)"}[m]] = (mu, sd)
                    for nm, (mu, sd) in preds.items():
                        e = y[qi] - (p_te[qi] + mu)
                        rec = {"fold": fo, "study": s, "k": k, "draw": dr, "model": nm, "sse": float(np.sum(e ** 2)), "n": len(qi),
                               "sst": float(np.sum((y[qi] - y[tr].mean()) ** 2)), "rmse_study": float(np.sqrt(np.mean(e ** 2)))}
                        if nm == "SR-GP":
                            hw = qn[k] * sd
                            rec.update(cover=float(np.mean(np.abs(e) <= hw)), width=float(2 * hw.mean()))
                        if nm == "offset":
                            rec.update(cover=float(np.mean(np.abs(e) <= qo[k])), width=float(2 * qo[k]))
                        rows.append(rec)
        print(f"{tgt} fold {fo} {time.time()-t0:.0f}s theta_full={np.round(th['full'],3)}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(RES / f"09_srgp_raw_{tgt}{SUF}.csv", index=False)
    pd.DataFrame(thetas).to_csv(RES / f"09_srgp_theta_{tgt}{SUF}.csv", index=False)
    vgs[0].to_csv(RES / f"09_variogram_{tgt}{SUF}.csv", index=False)
    S = R.groupby(["k", "model"]).apply(lambda z: pd.Series({
        "RMSE": np.sqrt(z.sse.sum() / z.n.sum()), "R2": 1 - z.sse.sum() / z.sst.sum(),
        "RMSE_study_median": z.rmse_study.median(),
        "coverage": np.average(z.cover.fillna(0), weights=z.n) if z.get("cover") is not None and z.cover.notna().any() else np.nan,
        "width": z.width.mean() if z.get("width") is not None and z.width.notna().any() else np.nan})).reset_index()
    S["target"] = tgt
    print(S.to_string(), flush=True)
    return S


if __name__ == "__main__":
    tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    pd.concat([run(t) for t in tg]).to_csv(RES / f"09_srgp_summary_{'_'.join(tg)}{SUF}.csv", index=False)
