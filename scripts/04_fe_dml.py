"""Within-study (study fixed-effect) double machine learning on a literature-compiled database.

For a treatment D and outcome Y:
  naive  : partially linear DML on pooled rows, cross-fitted by study folds (between-study confounding left in)
  FE-DML : within-study demeaning of Y, D and X (identifying studies only), then PLR-DML cross-fitted by study
  FE-OLS : linear regression with study fixed effects (classical meta-regression reference)
Cluster-robust (study) standard errors for DML via the influence function.
Heterogeneity: R-learner on FE-DML residuals, tau(Z) with Z = moderators.
Refutations: placebo (D permuted within study), random common cause, 80% study subsampling.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

RNG = np.random.default_rng(SEED)
NUIS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, random_state=SEED, n_jobs=6)

ORGANIC = ["ethanol", "methanol", "acetone", "other_organic", "water_organic"]


def treatments(df):
    """returns dict name -> (D series, row mask, excluded-from-X columns, is_binary)"""
    T = {}
    m = df.catalyst_class.isin(["none", "alkali_carbonate", "alkali_hydroxide"])
    T["Alkali catalyst (vs none)"] = ((df.catalyst_class != "none").astype(float), m,
                                     ["catalyst_class", "CatR"], True)
    m = df.solvent_class.isin(["water"] + ORGANIC)
    T["Organic (co)solvent (vs water)"] = (df.solvent_class.isin(ORGANIC).astype(float), m,
                                          ["solvent_class", "process_subtype"], True)
    m = df.atmosphere_class.isin(["reducing_H2", "inert_purge", "autogenic"])
    T["H2 atmosphere (vs inert/autogenic)"] = ((df.atmosphere_class == "reducing_H2").astype(float), m,
                                              ["atmosphere_class"], True)
    m = df["T"].between(150, 450)
    T["Temperature (+10 C)"] = (df["T"] / 10.0, m, ["T", "log_R0"], False)
    m = df["t"].notna()
    T["log residence time (+1 ln-unit)"] = (df["log_t"], m, ["log_t", "log_R0"], False)
    m = df["SL"].between(0, 60)
    T["Solid loading (+5 wt%)"] = (df["SL"] / 5.0, m, ["SL", "SBR"], False)
    return T


def demean(A, g):
    A = pd.DataFrame(A)
    return (A - A.groupby(g).transform("mean")).values


def plr_dml(Y, D, X, g, n_folds=5, binary=False):
    """cross-fitted PLR; returns theta, cluster SE, residuals"""
    idx = np.arange(len(Y))
    ug = np.unique(g); perm = dict(zip(ug, RNG.permutation(len(ug))))
    folds = list(GroupKFold(n_folds).split(idx, groups=np.array([perm[x] for x in g])))
    ry = np.zeros(len(Y)); rd = np.zeros(len(Y))
    for tr, te in folds:
        my = lgb.LGBMRegressor(**NUIS).fit(X[tr], Y[tr]); ry[te] = Y[te] - my.predict(X[te])
        md = lgb.LGBMRegressor(**NUIS).fit(X[tr], D[tr]); rd[te] = D[te] - md.predict(X[te])
    theta = np.sum(rd * ry) / np.sum(rd * rd)
    psi = (ry - theta * rd) * rd
    J = np.mean(rd * rd)
    cl = pd.Series(psi).groupby(g).sum().values
    se = np.sqrt(np.sum(cl ** 2)) / (len(Y) * J)
    return float(theta), float(se), ry, rd


def fe_ols(Y, D, X, g):
    Xd = demean(np.column_stack([D, X]), g)
    Yd = demean(Y.reshape(-1, 1), g).ravel()
    keep = np.std(Xd, axis=0) > 1e-9
    keep[0] = True
    res = sm.OLS(Yd, Xd[:, keep]).fit(cov_type="cluster", cov_kwds={"groups": g})
    return float(res.params[0]), float(res.bse[0])


def r_learner(ry, rd, Z):
    """tau(Z) minimising sum (ry - tau rd)^2 = weighted regression of ry/rd on Z with weights rd^2"""
    w = rd ** 2
    ok = w > 1e-6
    psi = np.where(ok, ry / np.where(ok, rd, 1), 0)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                          reg_lambda=5, verbose=-1, random_state=SEED, n_jobs=6)
    m.fit(Z[ok], psi[ok], sample_weight=w[ok])
    return m


def analyse(df, outcome):
    rows, cate_store = [], {}
    for name, (D, mask, excl, binary) in treatments(df).items():
        d = df[mask & df[outcome].notna() & D.notna()].reset_index(drop=True)
        Dv = D[mask & df[outcome].notna() & D.notna()].values.astype(float)
        nums = [c for c in NUM_FEED + NUM_PROC if c not in excl]
        cats = [c for c in CAT if c not in excl]
        X = design(d, cats=cats, nums=nums)
        X = X.fillna(X.median()).values     # median of covariates only (label-free); trees robust
        Y = d[outcome].values; g = d.study_id.values
        # identifying studies: within-study variation in D
        sd = pd.Series(Dv).groupby(g).transform("std").fillna(0).values
        idf = sd > 1e-9
        n_id_st = int(pd.Series(g[idf]).nunique())
        if n_id_st < 8:
            print(f"SKIP {name}: only {n_id_st} identifying studies", flush=True)
            rows.append({"outcome": outcome, "treatment": name, "n_rows": len(d), "n_identifying_studies": n_id_st,
                         "note": "not identifiable within studies (<8 studies)"})
            continue
        th_n, se_n, _, _ = plr_dml(Y, Dv, X, g, binary=binary)
        # FE-DML on identifying studies
        Yi, Di, Xi, gi = Y[idf], Dv[idf], X[idf], g[idf]
        Yd = demean(Yi.reshape(-1, 1), gi).ravel(); Dd = demean(Di.reshape(-1, 1), gi).ravel(); Xd = demean(Xi, gi)
        th_f, se_f, ry, rd = plr_dml(Yd, Dd, np.hstack([Xd, Xi]), gi)   # demeaned + levels as moderators of nuisance
        th_o, se_o = fe_ols(Yi, Di, Xi, gi)
        # refutations
        pls = []
        for b in range(10):                       # placebo distribution: D permuted within study
            Dp = pd.Series(Di).groupby(gi).transform(lambda s: s.sample(frac=1, random_state=int(RNG.integers(1e9))).values).values
            Dpd = demean(Dp.reshape(-1, 1), gi).ravel()
            pls.append(plr_dml(Yd, Dpd, np.hstack([Xd, Xi]), gi)[0])
        th_pl, se_pl = float(np.mean(pls)), float(np.std(pls))
        p_pl = float(np.mean(np.abs(pls) >= abs(th_f)))
        rc = RNG.normal(size=(len(Yi), 1))
        th_rc, _, _, _ = plr_dml(Yd, Dd, np.hstack([Xd, Xi, rc]), gi)
        sub = []
        ug = np.unique(gi)
        for b in range(10):
            keep_g = RNG.choice(ug, size=int(0.8 * len(ug)), replace=False)
            mk = np.isin(gi, keep_g)
            sub.append(plr_dml(Yd[mk], Dd[mk], np.hstack([Xd, Xi])[mk], gi[mk])[0])
        rows.append({"outcome": outcome, "treatment": name, "n_rows": len(d), "n_studies": int(d.study_id.nunique()),
                     "n_identifying_rows": int(idf.sum()), "n_identifying_studies": n_id_st,
                     "naive_DML": th_n, "naive_DML_se": se_n, "FE_DML": th_f, "FE_DML_se": se_f,
                     "FE_OLS": th_o, "FE_OLS_se": se_o, "placebo_mean": th_pl, "placebo_sd": se_pl, "placebo_p": p_pl,
                     "random_cause": th_rc, "subsample_mean": float(np.mean(sub)), "subsample_sd": float(np.std(sub))})
        print(rows[-1], flush=True)
        # heterogeneity on raw (level) moderators
        Zcols = ["Lignin", "Ash", "Cellulose", "T", "HC_feed", "OC_feed"]
        Zcols = [c for c in Zcols if c not in excl or c == "T"]   # T kept as moderator (local slope for the T treatment)
        Z = d.loc[idf, Zcols].astype(float)
        Z = Z.fillna(Z.median()).values
        mt = r_learner(ry, rd, Z)
        tau = mt.predict(Z)
        cate_store[name] = pd.DataFrame(np.column_stack([Z, tau, gi]), columns=Zcols + ["tau", "study_id"])
        cate_store[name].to_csv(RES / f"04_cate_{outcome}_{name.split(' ')[0]}.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    outs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    allr = pd.concat([analyse(df, o) for o in outs])
    allr.to_csv(RES / "04_fe_dml_effects_v2.csv", index=False)
    print(allr.to_string())
