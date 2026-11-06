"""Multi-output Study-Residual GP (MO-SR-GP) over product phases (oil, char, aqueous, gas).

Residuals r_ip = y_ip - f_p(x_i) from per-phase base learners (study-held-out OOF on training studies).
Within a study (rows i, j; phases p, q):
    cov(r_ip, r_jq) = B[p,q] * k(c_i, c_j) + S[p,q] * [i == j]
    k(c_i, c_j) = eta + exp(-||c_i - c_j||^2 / (2 l^2)) * rho^{m_ij}
B (between-phase lab-effect covariance) and S (same-run cross-phase noise) are PSD (Cholesky-parametrised);
rows may report any subset of phases. Hyper-parameters are fitted by exact-gradient (torch) maximisation of the
summed log-marginal likelihood over training studies. Independent per-phase SR-GPs (P = 1) are the ablation.

Scenarios on held-out studies (k anchor runs that report both oil and char):
  joint     anchors report every phase they measured; predict all phases of the remaining runs
  oil->char anchors report ONLY biocrude; predict solid/char of all remaining runs (cross-phase transfer)
  char->oil anchors report ONLY char; predict biocrude
Also: closure violations (oil + char > 100 wt%) of base vs adapted predictions.
"""
import sys, os, importlib.util, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import torch
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import *
spec = importlib.util.spec_from_file_location("srgp", HERE / "09_srgp.py"); S9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S9)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
RNG = np.random.default_rng(SEED + 13)
PH = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char", "Y_aq", "Y_gas"]
P = len(PH)
KS = [1, 3, 5]
CAP = 50
SUF = S9.SUF


# ------------------------------------------------------------------ model
class MOGP(torch.nn.Module):
    def __init__(self, P, var0):
        super().__init__()
        self.P = P
        self.LB = torch.nn.Parameter(torch.diag(torch.tensor(np.sqrt(0.7 * var0))).clone())
        self.LS = torch.nn.Parameter(torch.diag(torch.tensor(np.sqrt(0.3 * var0))).clone())
        self.log_l = torch.nn.Parameter(torch.tensor(np.log(1.5)))
        self.logit_rho = torch.nn.Parameter(torch.tensor(1.0))
        self.log_eta = torch.nn.Parameter(torch.tensor(np.log(0.5)))

    def mats(self):
        tril = torch.tril(torch.ones(self.P, self.P))
        LB = self.LB * tril; LS = self.LS * tril
        return LB @ LB.T, LS @ LS.T + 1e-4 * torch.eye(self.P)

    def kc(self, Ci, Cj, Mi, Mj):
        d2 = ((Ci[:, None, :] - Cj[None, :, :]) ** 2).sum(-1)
        mm = (Mi[:, None, :] != Mj[None, :, :]).sum(-1).double()
        rho = torch.sigmoid(self.logit_rho); l = torch.exp(self.log_l)
        return torch.exp(self.log_eta) + torch.exp(-d2 / (2 * l ** 2)) * rho ** mm

    def cov(self, Ca, Ma, Ia, Pa, Cb, Mb, Ib, Pb, same_set=False):
        B, Sm = self.mats()
        K = B[Pa][:, Pb] * self.kc(Ca, Cb, Ma, Mb)[Ia][:, Ib]
        if same_set:
            K = K + Sm[Pa][:, Pb] * (Ia[:, None] == Ib[None, :]).double()
        return K

    def nlml(self, pk):
        """batched negative log marginal likelihood over padded studies (pk from pack())"""
        B, Sm = self.mats()
        C, M, I, Pp, r, mask = pk["C"], pk["M"], pk["I"], pk["P"], pk["r"], pk["mask"]
        d2 = ((C[:, :, None, :] - C[:, None, :, :]) ** 2).sum(-1)
        mm = (M[:, :, None, :] != M[:, None, :, :]).sum(-1).double()
        rho = torch.sigmoid(self.logit_rho); l = torch.exp(self.log_l)
        kc = torch.exp(self.log_eta) + torch.exp(-d2 / (2 * l ** 2)) * rho ** mm
        b = torch.arange(C.shape[0])[:, None, None]
        Kc = kc[b, I[:, :, None], I[:, None, :]]
        K = B[Pp[:, :, None], Pp[:, None, :]] * Kc + Sm[Pp[:, :, None], Pp[:, None, :]] * (I[:, :, None] == I[:, None, :]).double()
        m2 = mask[:, :, None] & mask[:, None, :]
        K = torch.where(m2, K, torch.zeros_like(K)) + torch.diag_embed(torch.where(mask, torch.full_like(r, 1e-6), torch.ones_like(r)))
        L = torch.linalg.cholesky(K)
        a = torch.cholesky_solve(r[..., None], L)
        return 0.5 * (r[..., None] * a).sum() + torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum()


def pack(studies):
    S_ = len(studies); nmax = max(len(x[0]) for x in studies); Nmax = max(len(x[4]) for x in studies)
    C = torch.zeros(S_, nmax, studies[0][0].shape[1]); M = torch.zeros(S_, nmax, studies[0][1].shape[1], dtype=torch.int64)
    I = torch.zeros(S_, Nmax, dtype=torch.int64); Pp = torch.zeros(S_, Nmax, dtype=torch.int64)
    r = torch.zeros(S_, Nmax); mask = torch.zeros(S_, Nmax, dtype=torch.bool)
    for j, (c, m, i, pp, rr) in enumerate(studies):
        C[j, :len(c)] = c; M[j, :len(m)] = m; I[j, :len(i)] = i; Pp[j, :len(pp)] = pp; r[j, :len(rr)] = rr; mask[j, :len(rr)] = True
    return {"C": C, "M": M, "I": I, "P": Pp, "r": r, "mask": mask}


def fit(model, studies, iters=40):
    pk = pack(studies)
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); loss = model.nlml(pk); loss.backward(); return loss
    opt.step(closure)
    return model


# ------------------------------------------------------------------ data helpers
def tens(x, dtype=torch.float64):
    return torch.tensor(np.asarray(x), dtype=dtype)


def study_tensor(rows, R, C, M, phases):
    """entries (row, phase) observed among `rows`; R is n x P residual matrix with NaN for unobserved"""
    I, Pp, r = [], [], []
    for li, i in enumerate(rows):
        for p in phases:
            if not np.isnan(R[i, p]):
                I.append(li); Pp.append(p); r.append(R[i, p])
    return (tens(C[rows]), tens(M[rows], torch.int64), tens(I, torch.int64), tens(Pp, torch.int64), tens(r))


def entries(rws, phases, R=None):
    I, Pp, r = [], [], []
    for i in rws:
        for p in phases:
            if R is None or not np.isnan(R[i, p]):
                I.append(i); Pp.append(p); r.append(np.nan if R is None else R[i, p])
    return np.array(I, dtype=int), np.array(Pp, dtype=int), np.array(r, dtype=float)


def mo_predict(model, ai, qi, a_ph, q_ph, Rfull, Ct, Mt, single=False):
    """GP posterior mean of query residuals (rows qi, phases q_ph) given anchor residuals (rows ai, phases a_ph)."""
    Ia, Pa, ra = entries(ai, a_ph, Rfull)
    Iq, Pq, _ = entries(qi, q_ph)
    if len(ra) == 0 or len(Iq) == 0:
        return Iq, Pq, np.zeros(len(Iq))
    pa = np.zeros_like(Pa) if single else Pa
    pq = np.zeros_like(Pq) if single else Pq
    with torch.no_grad():
        B, Sm = model.mats()
        Pa_t, Pq_t = tens(pa, torch.int64), tens(pq, torch.int64)
        Ca, Ma, Cq, Mq = Ct[Ia], Mt[Ia], Ct[Iq], Mt[Iq]
        same = tens((Ia[:, None] == Ia[None, :]).astype(float))
        Kaa = B[Pa_t][:, Pa_t] * model.kc(Ca, Ca, Ma, Ma) + Sm[Pa_t][:, Pa_t] * same + 1e-6 * torch.eye(len(Ia))
        Kqa = B[Pq_t][:, Pa_t] * model.kc(Cq, Ca, Mq, Ma)
        L = torch.linalg.cholesky(Kaa)
        mu = (Kqa @ torch.cholesky_solve(tens(ra)[:, None], L)).ravel().numpy()
    return Iq, Pq, mu


def run():
    d = load()
    ymat = d[PH].values.astype(float)
    keep = ~np.all(np.isnan(ymat), axis=1)
    d = d[keep].reset_index(drop=True); ymat = ymat[keep]
    X = design(d).values; g = d.study_id.values
    Craw = d[S9.COND].astype(float).values
    Mcodes = np.column_stack([pd.factorize(d[c].astype(str))[0] for c in S9.MATCH])
    io, ic = PH.index("Y_oil"), PH.index("Y_char")
    rows = []
    for fo, (tr, te) in enumerate(splits(d, "group")):
        t0 = time.time()
        C = np.nan_to_num((Craw - np.nanmean(Craw[tr], 0)) / (np.nanstd(Craw[tr], 0) + 1e-9))
        dtr = d.iloc[tr].reset_index(drop=True)
        inner = splits(dtr, "group", seed=SEED + 3)
        Rin = np.full_like(ymat, np.nan); Pte = np.full_like(ymat, np.nan)
        for p in range(P):
            ob = ~np.isnan(ymat[:, p])
            trp = tr[ob[tr]]
            # inner OOF residuals on training rows of this phase
            oofp = np.full(len(d), np.nan)
            for itr, ite in inner:
                a = tr[itr][ob[tr[itr]]]; b = tr[ite][ob[tr[ite]]]
                if len(b) == 0: continue
                oofp[b] = S9.make_base().fit(X[a], ymat[a, p]).predict(X[b])
            Rin[trp, p] = ymat[trp, p] - oofp[trp]
            Pte[te, p] = S9.make_base().fit(X[trp], ymat[trp, p]).predict(X[te])
        # training studies -> tensors (capped)
        st_all, st_single = [], {p: [] for p in range(P)}
        for s in np.unique(g[tr]):
            idx = np.where(g == s)[0]
            if len(idx) < 2: continue
            if len(idx) > CAP: idx = RNG.choice(idx, CAP, replace=False)
            st_all.append(study_tensor(idx, Rin, C, Mcodes, range(P)))
            for p in range(P):
                ii = idx[~np.isnan(Rin[idx, p])]
                if len(ii) >= 2:
                    st_single[p].append(study_tensor(ii, Rin[:, [p]], C, Mcodes, [0]))
        var0 = np.nanvar(Rin[tr], axis=0)
        mo = fit(MOGP(P, var0), st_all)
        ind = {p: fit(MOGP(1, var0[[p]]), st_single[p]) for p in range(P)}
        with torch.no_grad():
            B, Sm = mo.mats(); Bc = B.numpy(); Dg = np.sqrt(np.diag(Bc))
            corrB = Bc / np.outer(Dg, Dg)
        Ct, Mt = tens(C), tens(Mcodes, torch.int64)
        # ------------------------------------------------ test episodes
        for s in np.unique(g[te]):
            idx = np.where(g == s)[0]
            both = idx[~np.isnan(ymat[idx, io]) & ~np.isnan(ymat[idx, ic])]
            for k in KS:
                if len(both) < k + 2: continue
                for dr in range(15):
                    ai = RNG.choice(both, k, replace=False); qi = np.setdiff1d(idx, ai)
                    Ytrue = ymat[qi]; base = Pte[qi]
                    Ra = ymat[ai] - Pte[ai]
                    out = {}
                    out["base"] = base.copy()
                    out["offset"] = base + np.where(np.isnan(np.nanmean(Ra, 0)), 0, np.nanmean(Ra, 0))[None, :]

                    Rfull = ymat - Pte
                    pos = {v: i for i, v in enumerate(qi)}

                    def apply(A, Iq, Pq, mu, only=None):
                        A = A.copy()
                        for i_, p_, m_ in zip(Iq, Pq, mu):
                            if only is None or p_ == only:
                                A[pos[i_], p_] += m_
                        return A
                    Iq, Pq, mu = mo_predict(mo, ai, qi, range(P), range(P), Rfull, Ct, Mt)
                    out["MO-SR-GP"] = apply(base, Iq, Pq, mu)
                    A = base.copy()
                    for p in range(P):
                        Iq, Pq, mu = mo_predict(ind[p], ai, qi, [p], [p], Rfull, Ct, Mt, single=True)
                        A = apply(A, Iq, Pq, mu)
                    out["indep SR-GP"] = A
                    Iq, Pq, mu = mo_predict(mo, ai, qi, [io], [ic], Rfull, Ct, Mt)
                    out["MO oil->char"] = apply(base, Iq, Pq, mu)
                    Iq, Pq, mu = mo_predict(mo, ai, qi, [ic], [io], Rfull, Ct, Mt)
                    out["MO char->oil"] = apply(base, Iq, Pq, mu)
                    for nm, A in out.items():
                        for p in range(P):
                            ok = ~np.isnan(Ytrue[:, p])
                            if ok.sum() == 0: continue
                            e = Ytrue[ok, p] - A[ok, p]
                            rows.append({"fold": fo, "study": s, "k": k, "draw": dr, "model": nm, "phase": PH[p],
                                         "sse": float(np.sum(e ** 2)), "n": int(ok.sum()),
                                         "sst": float(np.sum((Ytrue[ok, p] - np.nanmean(ymat[tr, p])) ** 2)),
                                         "closure_viol": float(np.mean((A[:, io] + A[:, ic]) > 100.0)) if p == 0 else np.nan})
        print(f"fold {fo} {time.time()-t0:.0f}s corrB(oil,char)={corrB[io, ic]:.2f}", flush=True)
        print(np.round(corrB, 2), flush=True)
    R = pd.DataFrame(rows); R.to_csv(RES / f"13_mo_srgp_raw{SUF}.csv", index=False)
    Sm = R.groupby(["k", "model", "phase"]).apply(lambda z: pd.Series({
        "RMSE": np.sqrt(z.sse.sum() / z.n.sum()), "R2": 1 - z.sse.sum() / z.sst.sum(), "n_q": z.n.sum()})).reset_index()
    Sm.to_csv(RES / f"13_mo_srgp_summary{SUF}.csv", index=False)
    print(Sm.pivot_table(index=["phase", "k"], columns="model", values="R2").round(3).to_string())
    cv = R[R.phase == PH[0]].groupby(["k", "model"]).closure_viol.mean().unstack().round(4)
    print("closure violation share (oil+char>100):\n", cv)


if __name__ == "__main__":
    run()
