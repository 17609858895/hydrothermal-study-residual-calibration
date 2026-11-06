"""Active anchor selection with SR-GP (label-free experimental design for a new laboratory).

For each held-out study, the planned run conditions are known before any yield is measured.
Greedy integrated-variance reduction (IVR): choose the next anchor that minimises the summed SR-GP
posterior variance over the study's remaining planned runs. Compare with random anchors (20 draws)
and with a 'space-filling' baseline (greedy max-min condition distance).
"""
import sys, importlib.util, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.linalg import cho_factor, cho_solve
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import *
spec = importlib.util.spec_from_file_location("srgp", HERE / "09_srgp.py"); S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

RNG = np.random.default_rng(SEED + 11)
KS = [1, 2, 3, 4, 5]


def post_var_sum(C, M, th, A, Q):
    if len(A) == 0:
        return np.inf
    Kaa = S.kern(C[A], C[A], M[A], M[A], th) + (th[4] ** 2 + 1e-6) * np.eye(len(A))
    Kqa = S.kern(C[Q], C[A], M[Q], M[A], th)
    cf = cho_factor(Kaa, lower=True)
    return -np.einsum("ij,ji->i", Kqa, cho_solve(cf, Kqa.T)).sum()


def greedy_ivr(C, M, th, idx, k):
    A = []
    for _ in range(k):
        best, bv = None, np.inf
        for c in idx:
            if c in A: continue
            AA = A + [c]; Q = [q for q in idx if q not in AA]
            v = post_var_sum(C, M, th, np.array(AA), np.array(Q))
            if v < bv: bv, best = v, c
        A.append(best)
    return np.array(A)


def greedy_maxmin(C, M, idx, k):
    D = np.sqrt(((C[idx][:, None] - C[idx][None]) ** 2).sum(-1)) + (M[idx][:, None] != M[idx][None]).sum(-1)
    # start from the run closest to the study's centroid
    cen = np.argmin(((C[idx] - C[idx].mean(0)) ** 2).sum(1)); A = [cen]
    while len(A) < k:
        scores = D[:, A].min(1)
        scores[np.asarray(A, dtype=int)] = -np.inf
        A.append(int(np.argmax(scores)))
    return idx[np.array(A)]


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values; g = d.study_id.values
    Craw = d[S.COND].astype(float).values; M = d[S.MATCH].astype(str).values
    rows = []
    for fo, (tr, te) in enumerate(splits(d, "group")):
        t0 = time.time()
        C = np.nan_to_num((Craw - np.nanmean(Craw[tr], 0)) / (np.nanstd(Craw[tr], 0) + 1e-9))
        dtr = d.iloc[tr].reset_index(drop=True)
        r_in = np.zeros(len(d)); r_in[tr] = y[tr] - S.oof(X[tr], y[tr], splits(dtr, "group", seed=SEED + 3))
        p_te = np.zeros(len(d)); p_te[te] = lgb.LGBMRegressor(**S.LGB).fit(X[tr], y[tr]).predict(X[te])
        th = S.fit_theta(C, M, r_in, g, np.unique(g[tr]), "full")
        for s in np.unique(g[te]):
            idx = np.where(g == s)[0]
            if len(idx) < 8: continue
            idx_c = idx if len(idx) <= 60 else RNG.choice(idx, 60, replace=False)   # candidate pool cap
            for k in KS:
                strategies = {"IVR (active)": [greedy_ivr(C, M, th, list(idx_c), k)],
                              "max-min distance": [greedy_maxmin(C, M, idx_c, k)],
                              "random": [RNG.choice(idx, k, replace=False) for _ in range(20)]}
                for nm, draws in strategies.items():
                    for ai in draws:
                        qi = np.setdiff1d(idx, ai)
                        mu, sd = S.posterior(C[ai], M[ai], y[ai] - p_te[ai], C[qi], M[qi], th, "full")
                        e = y[qi] - (p_te[qi] + mu)
                        rows.append({"fold": fo, "study": s, "k": k, "strategy": nm, "sse": float(np.sum(e ** 2)),
                                     "n": len(qi), "sst": float(np.sum((y[qi] - y[tr].mean()) ** 2)),
                                     "rmse_study": float(np.sqrt(np.mean(e ** 2)))})
        print(f"{tgt} fold {fo} {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); R.to_csv(RES / f"11_active_raw_{tgt}.csv", index=False)
    # per-study averages first (random has 20 draws), then pooled
    st = R.groupby(["study", "k", "strategy"]).agg(sse=("sse", "mean"), n=("n", "mean"), sst=("sst", "mean"),
                                                   rmse=("rmse_study", "mean")).reset_index()
    Sm = st.groupby(["k", "strategy"]).apply(lambda z: pd.Series({
        "RMSE": np.sqrt(z.sse.sum() / z.n.sum()), "R2": 1 - z.sse.sum() / z.sst.sum(),
        "RMSE_study_median": z.rmse.median(), "n_studies": len(z)})).reset_index()
    Sm["target"] = tgt
    # paired win rate active vs random
    pv = st.pivot_table(index=["study", "k"], columns="strategy", values="rmse").reset_index()
    win = pv.groupby("k").apply(lambda z: float(np.mean(z["IVR (active)"] < z["random"]))).rename("win_rate_vs_random")
    Sm = Sm.merge(win, on="k")
    print(Sm.to_string(), flush=True)
    return Sm


if __name__ == "__main__":
    tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    pd.concat([run(t) for t in tg]).to_csv(RES / f"11_active_summary_{'_'.join(tg)}.csv", index=False)
