"""Derived summaries for the extended composite figures (2026-11-13). No reported model is changed.

A. Corpus structure: records per source study, phase co-reporting, feedstock family by process.
B. Per-study error of the pooled LightGBM under random-row and held-out-study validation
   (same predictors and configuration as 02_quick_baseline.py, composition and process predictors).
C. Paired RMSE gain of residual correction over the constant offset with study-bootstrap intervals,
   and the share of studies with lower residual-correction RMSE, from the stored 09 episodes.
D. Four-phase element-wise RMSE by anchor budget from the stored 13 episodes.
E. Interval adaptivity: query RMSE of residual-correction episodes grouped by interval-width quintile.
F. Variance-selected versus random anchors per study from the stored 11 episodes.
G. Three illustrative held-out biocrude studies at the quartiles of the three-anchor RMSE ratio
   (residual correction / constant offset; random-anchor episodes of 09); each is refitted exactly as in
   09_srgp.py and its three anchors are chosen by integrated-variance reduction (11_active_anchor.py).
H. Reported four-phase yield sums for records with all four phases.
Outputs: results/20_*.csv and results/20_panel_support.json
"""
import sys, json, importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import *

rng = np.random.default_rng(SEED + 20)
OUT = {}
PH = ["Y_oil", "Y_char", "Y_aq", "Y_gas"]
SIX = ["Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "HHV_oil"]


def part_a():
    d = load().reset_index(drop=True)
    g = d.groupby("study_id")
    S = pd.DataFrame({"n": g.size(), "n_oil": g.Y_oil.count(), "n_char": g.Y_char.count(), "year": g.year_num.first(),
                      "subtype": g.process_subtype.agg(lambda s: s.value_counts().index[0]),
                      "n_feedstock": g.Feedstock.nunique(), "n_T": g["T"].nunique()}).reset_index()
    S.to_csv(RES / "20_study_sizes.csv", index=False)
    n = np.sort(S.n.values)[::-1]
    OUT["study_size"] = {"median": float(np.median(n)), "max": int(n[0]), "share_top10": float(n[:10].sum() / n.sum()),
                         "share_top20pct_studies": float(n[: int(np.ceil(0.2 * len(n)))].sum() / n.sum()),
                         "studies_below_10": int((n < 10).sum())}
    key = d[PH].notna().apply(lambda r: "+".join(p for p, v in zip(["oil", "char", "aq", "gas"], r) if v) or "none", axis=1)
    C = key.value_counts().rename("records").to_frame()
    C["studies"] = d.groupby(key).study_id.nunique()
    C.reset_index(names="combo").to_csv(RES / "20_phase_combos.csv", index=False)
    F = d.groupby(["Family_std", "process_subtype"]).size().unstack(fill_value=0)
    F["total"] = F.sum(1); F.sort_values("total", ascending=False).reset_index().to_csv(RES / "20_family_process.csv", index=False)
    A = pd.DataFrame({"response": SIX, "records": [int(d[t].notna().sum()) for t in SIX],
                      "studies": [int(d.loc[d[t].notna(), "study_id"].nunique()) for t in SIX]})
    A.to_csv(RES / "20_response_availability.csv", index=False)


def part_b():
    d0 = load(); rows = []
    for tgt in ["Y_oil", "Y_char"]:
        d = d0[d0[tgt].notna()].reset_index(drop=True)
        X = design(d, cats=CAT_CHEM).values; y = d[tgt].values
        P = {}
        for kind in ["random", "group"]:
            oof = np.zeros(len(d))
            for tr, te in splits(d, kind):
                m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED, n_jobs=8)
                oof[te] = m.fit(X[tr], y[tr]).predict(X[te])
            P[kind] = oof
        z = pd.DataFrame({"study_id": d.study_id, "e_random": y - P["random"], "e_group": y - P["group"]})
        s = z.groupby("study_id").agg(n=("e_random", "size"), rmse_random=("e_random", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                                      rmse_group=("e_group", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                                      bias_group=("e_group", "mean")).reset_index()
        s["target"] = tgt; rows.append(s)
        OUT[f"per_study_error_{tgt}"] = {"median_rmse_random": float(s.rmse_random.median()), "median_rmse_group": float(s.rmse_group.median()),
                                         "share_studies_group_worse": float(np.mean(s.rmse_group > s.rmse_random)), "studies": int(len(s))}
    pd.concat(rows).to_csv(RES / "20_per_study_error_split.csv", index=False)


def part_c(nb=2000):
    rows = []
    # The five-anchor primary-yield intervals reuse the resampling stream of the supplementary table
    # (seed 2041, biocrude then char) so that figure and table report identical limits.
    table_rng = np.random.default_rng(2041)
    order = [("Y_oil", 5), ("Y_char", 5)] + [(t, k) for t in SIX for k in (1, 3, 5) if not (t in ("Y_oil", "Y_char") and k == 5)]
    cache = {}
    for tgt, k in order:
        if tgt not in cache:
            r = pd.read_csv(RES / f"09_srgp_raw_{tgt}.csv"); cache[tgt] = r[r.model.isin(["offset", "SR-GP"])]
        z = cache[tgt]; z = z[z.k == k]
        if True:
            A = z.groupby(["study", "model"])[["sse", "n"]].sum().unstack("model")
            so, sg, no, ng = A[("sse", "offset")].values, A[("sse", "SR-GP")].values, A[("n", "offset")].values, A[("n", "SR-GP")].values
            gain = lambda ix: np.sqrt(so[ix].sum() / no[ix].sum()) - np.sqrt(sg[ix].sum() / ng[ix].sum())
            full = gain(np.arange(len(so)))
            use = table_rng if (tgt in ("Y_oil", "Y_char") and k == 5) else rng
            sampled = use.integers(0, len(so), size=(nb, len(so)))
            bs = np.sqrt(so[sampled].sum(1) / no[sampled].sum(1)) - np.sqrt(sg[sampled].sum(1) / ng[sampled].sum(1))
            st = z.groupby(["study", "model"]).rmse_study.mean().unstack()
            rows.append({"target": tgt, "k": int(k), "studies": int(len(so)), "rmse_offset": float(np.sqrt(so.sum() / no.sum())),
                         "rmse_srgp": float(np.sqrt(sg.sum() / ng.sum())), "gain": float(full),
                         "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                         "share_studies_srgp_lower": float(np.mean(st["SR-GP"] < st["offset"])),
                         "median_ratio": float(np.median(st["SR-GP"] / st["offset"]))})
    pd.DataFrame(rows).to_csv(RES / "20_rmse_gain_ci.csv", index=False)


def part_d():
    m = pd.read_csv(RES / "13_mo_srgp_raw.csv")
    m = m[m.model.isin(["base", "offset", "indep SR-GP", "MO-SR-GP"])]
    # element-wise RMSE over the four phases of one episode, averaged over draws within a study, median over studies
    e = m.groupby(["study", "k", "draw", "model"]).agg(sse=("sse", "sum"), n=("n", "sum"), phases=("phase", "nunique")).reset_index()
    e = e[e.phases == 4]; e["rmse"] = np.sqrt(e.sse / e.n)
    s = e.groupby(["study", "k", "model"]).rmse.mean().reset_index()
    rows = []
    for (k, model), z in s.groupby(["k", "model"]):
        v = z.rmse.values
        bs = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
        rows.append({"k": int(k), "model": model, "studies": int(len(v)), "median_rmse": float(np.median(v)),
                     "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))})
    pd.DataFrame(rows).to_csv(RES / "20_fourphase_rmse.csv", index=False)


def part_e():
    rows = []
    for tgt in ["Y_oil", "Y_char"]:
        r = pd.read_csv(RES / f"09_srgp_raw_{tgt}.csv"); r = r[(r.k == 5) & r.width.notna()]
        g = r[r.model == "SR-GP"].copy()
        g["q"] = pd.qcut(g.width, 5, labels=False)
        for qq, z in g.groupby("q"):
            rows.append({"target": tgt, "model": "SR-GP", "quintile": int(qq) + 1, "width": float(z.width.mean()),
                         "rmse": float(np.sqrt(z.sse.sum() / z.n.sum())), "coverage": float((z.cover * z.n).sum() / z.n.sum()),
                         "episodes": int(len(z))})
        o = r[r.model == "offset"]
        rows.append({"target": tgt, "model": "offset", "quintile": 0, "width": float(o.width.mean()),
                     "rmse": float(np.sqrt(o.sse.sum() / o.n.sum())), "coverage": float((o.cover * o.n).sum() / o.n.sum()), "episodes": int(len(o))})
        w = r.groupby(["study", "model"]).width.mean().unstack()
        w["target"] = tgt; w.reset_index().to_csv(RES / f"20_study_width_{tgt}.csv", index=False)
    pd.DataFrame(rows).to_csv(RES / "20_width_adaptivity.csv", index=False)


def part_f():
    rows = []
    for tgt in ["Y_oil", "Y_char"]:
        a = pd.read_csv(RES / f"11_active_raw_{tgt}.csv")
        s = a.groupby(["study", "k", "strategy"]).rmse_study.mean().unstack("strategy").reset_index()
        s["diff_random_minus_ivr"] = s["random"] - s["IVR (active)"]; s["target"] = tgt
        rows.append(s)
    pd.concat(rows).to_csv(RES / "20_active_per_study.csv", index=False)


def part_g(tgt="Y_oil", k=3, min_runs=12):
    def mod(name, file):
        spec = importlib.util.spec_from_file_location(name, HERE / file); m_ = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m_); return m_
    S = mod("srgp", "09_srgp.py"); A = mod("act", "11_active_anchor.py")
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values; g = d.study_id.values
    r = pd.read_csv(RES / f"09_srgp_raw_{tgt}.csv"); r = r[r.k == k]
    st = r.groupby(["study", "model"]).rmse_study.mean().unstack()
    st["ratio"] = st["SR-GP"] / st["offset"]
    nrun = pd.Series(g).value_counts()
    cand = st[st.index.map(nrun) >= min_runs].copy()
    picks = {}
    for name, qv in [("lower quartile", 0.25), ("median", 0.50), ("upper quartile", 0.75)]:
        target = cand.ratio.quantile(qv)
        picks[name] = int((cand.ratio - target).abs().idxmin())
    Craw = d[S.COND].astype(float).values; M = d[S.MATCH].astype(str).values
    folds = splits(d, "group"); cases = []; info = {"selection_set": int(len(cand)), "quartiles": {}}
    for name, s in picks.items():
        fo = [i for i, (_, te) in enumerate(folds) if s in set(g[te])][0]
        tr, te = folds[fo]
        mu_c = np.nanmean(Craw[tr], 0); sd_c = np.nanstd(Craw[tr], 0) + 1e-9
        C = np.nan_to_num((Craw - mu_c) / sd_c)
        inner = splits(d.iloc[tr].reset_index(drop=True), "group", seed=SEED + 3)
        r_in = np.zeros(len(d)); r_in[tr] = y[tr] - S.oof(X[tr], y[tr], inner)
        th = S.fit_theta(C, M, r_in, g, np.unique(g[tr]), "full")
        idx = np.where(g == s)[0]
        p = S.make_base().fit(X[tr], y[tr]).predict(X[idx])
        a = A.greedy_ivr(C, M, th, list(idx), k)
        am = np.isin(idx, a); ra = y[idx][am] - p[am]
        mu, sd = S.posterior(C[idx][am], M[idx][am], ra, C[idx], M[idx], th, "full")
        z = pd.DataFrame({"case": name, "study": s, "y": y[idx], "pooled": p, "offset": p + ra.mean(), "srgp": p + mu, "sd": sd, "anchor": am})
        cases.append(z); qm = ~am
        info["quartiles"][name] = {"study": s, "runs": int(len(idx)), "ratio_random_anchor_episodes": float(cand.ratio.loc[s]),
                                   **{f"rmse_{c}": float(np.sqrt(np.mean((z.y[qm] - z[c][qm]) ** 2))) for c in ["pooled", "offset", "srgp"]}}
    pd.concat(cases).to_csv(RES / "20_example_studies.csv", index=False)
    OUT["example_studies"] = info


def part_h():
    d = load(); z = d.dropna(subset=PH)
    pd.DataFrame({"study_id": z.study_id, "sum4": z[PH].sum(1)}).to_csv(RES / "20_closure_sums.csv", index=False)
    s4 = z[PH].sum(1)
    OUT["closure_sums"] = {"records": int(len(z)), "studies": int(z.study_id.nunique()),
                           "share_within_1": float(np.mean((s4 - 100).abs() < 1)), "share_within_5": float(np.mean((s4 - 100).abs() < 5))}


if __name__ == "__main__":
    only = sys.argv[1:] or list("abcdefgh")
    for p in only:
        globals()[f"part_{p}"](); print("done", p, flush=True)
    f = RES / "20_panel_support.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(OUT); f.write_text(json.dumps(old, indent=1))
    print(json.dumps(OUT, indent=1))
