"""Can reported metadata explain the laboratory effect?

1) study effects u_s = mean OOF residual of the pooled study-held-out model (shrunk by n/(n+3)).
2) study-level descriptors (label-free): year, n runs, extraction/work-up class, yield basis,
   reactor class, process subtype share, solvent share, catalyst share, mean T, mean feedstock C/Ash.
3) study-grouped (here: leave-studies-out) CV R2 of predicting u_s from descriptors (LightGBM + Ridge).
Also: variance decomposition of y into between-study / within-study-explainable / within-transferable.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *


def mode(s):
    return s.value_counts().index[0]


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    o = pd.read_csv(RES / f"03_oof_{tgt}.csv")
    assert len(o) == len(d)
    d["res"] = o["y"].values - o["P"].values
    agg = d.groupby("study_id").agg(u=("res", "mean"), n=("res", "size"), year=("year_num", "first"),
                                    T_mean=("T", "mean"), C_mean=("C", "mean"), Ash_mean=("Ash", "mean"),
                                    extraction=("extraction_class", mode), basis=("basis_class", mode),
                                    reactor=("reactor_class", mode), subtype=("process_subtype", mode),
                                    solvent=("solvent_class", mode), catalyst=("catalyst_class", mode),
                                    family=("Family_std", mode))
    agg["u_shrunk"] = agg.u * agg.n / (agg.n + 3)
    Z = pd.get_dummies(agg.drop(columns=["u", "u_shrunk"]), dtype=float)
    Z["log_n"] = np.log(agg.n)
    yv = agg.u_shrunk.values
    out = {"target": tgt, "n_studies": len(agg), "var_u": float(np.var(yv))}
    for nm in ["Ridge", "LightGBM"]:
        p = np.zeros(len(yv))
        for rep in range(5):
            for tr, te in KFold(5, shuffle=True, random_state=rep).split(Z):
                if nm == "Ridge":
                    sc = StandardScaler().fit(Z.iloc[tr].fillna(Z.iloc[tr].median()))
                    m = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(sc.transform(Z.iloc[tr].fillna(Z.iloc[tr].median())), yv[tr],
                                                                   sample_weight=agg.n.values[tr])
                    p[te] += m.predict(sc.transform(Z.iloc[te].fillna(Z.iloc[tr].median()))) / 5
                else:
                    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                                          verbose=-1, random_state=rep, n_jobs=4).fit(Z.iloc[tr], yv[tr], sample_weight=agg.n.values[tr])
                    p[te] += m.predict(Z.iloc[te]) / 5
        w = agg.n.values
        out[f"R2_u_from_metadata_{nm}"] = float(1 - np.sum(w * (yv - p) ** 2) / np.sum(w * (yv - np.average(yv, weights=w)) ** 2))
    agg.to_csv(RES / f"10_study_effects_{tgt}.csv")
    return out


if __name__ == "__main__":
    res = {t: run(t) for t in ["Y_oil", "Y_char"]}
    # variance budget from 07 diagnostics + 03 ICC
    d7 = json.load(open(RES / "07_within_diag.json")); d3 = json.load(open(RES / "03_summary_Y_oil_Y_char.json"))
    for t in res:
        w = d7[t]["within_var_share"]
        res[t]["budget"] = {"between_study": 1 - w,
                            "within_transferable(new study)": w * max(d7[t]["FE_withinR2_group"], 0),
                            "within_study_specific": w * (d7[t]["FE_withinR2_random"] - max(d7[t]["FE_withinR2_group"], 0)),
                            "within_unexplained": w * (1 - d7[t]["FE_withinR2_random"])}
        res[t]["ICC_raw"] = d3[t]["ICC_raw_y"]; res[t]["ICC_resid"] = d3[t]["ICC_resid_P"]
    print(json.dumps(res, indent=1))
    json.dump(res, open(RES / "10_lab_effect.json", "w"), indent=1)
