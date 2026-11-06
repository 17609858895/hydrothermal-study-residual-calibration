"""Is the negative bio-oil / char laboratory-effect correlation just a closure artefact?

Study effects b_s,phase = mean study-held-out residual of a pooled LightGBM (GroupKFold by study) over the
study's runs (studies with >= MIN_RUNS runs reporting the phase). Then
 1. empirical corr(b_oil, b_char) over all studies, and split by how the study closes its mass balance:
      forced closure  : median |oil+char+aq+gas - 100| < 1 wt% (one phase typically by difference)
      open balance    : median |sum - 100| >= 3 wt%, or aq/gas not reported at all
 2. closure null: independent phase effects with the observed variances, projected onto the sum-zero
    plane (P V P, P = I - 11'/4); the oil-char correlation this alone produces.
 3. permutation null: shuffle char effects across studies (corr ~ 0) - sanity bound.
Study-bootstrap 95% intervals for the correlations.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

PH = ["Y_oil", "Y_char", "Y_aq", "Y_gas"]
MIN_RUNS = 3
RNG = np.random.default_rng(SEED + 15)
LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED, n_jobs=8)

d = load().reset_index(drop=True)
X = design(d).values
res = pd.DataFrame(index=d.index, columns=PH, dtype=float)
for p in PH:
    m = d[p].notna().values
    dd = d[m].reset_index()
    oof = np.zeros(m.sum())
    for tr, te in splits(dd, "group"):
        oof[te] = lgb.LGBMRegressor(**LGB).fit(X[m][tr], dd[p].values[tr]).predict(X[m][te])
    res.loc[dd["index"].values, p] = dd[p].values - oof
d["sum4"] = d[PH].sum(axis=1, min_count=4)

rows = []
for s, z in d.groupby("study_id"):
    r = res.loc[z.index]
    rec = {"study": s, "n": len(z)}
    for p in PH:
        ok = r[p].notna()
        rec[p] = r.loc[ok, p].mean() if ok.sum() >= MIN_RUNS else np.nan
    dev = (z.sum4 - 100).abs().median()
    rec["closure_dev"] = dev
    rec["has_aq_gas"] = bool(z[["Y_aq", "Y_gas"]].notna().any(axis=None))
    rows.append(rec)
B = pd.DataFrame(rows)
B["balance"] = np.where(B.closure_dev < 1, "forced", np.where((B.closure_dev >= 3) | (~B.has_aq_gas), "open", "near"))


def corr_ci(b, a="Y_oil", c="Y_char", nb=2000):
    b = b[[a, c]].dropna()
    r0 = np.corrcoef(b[a], b[c])[0, 1]
    bs = [np.corrcoef(*b.iloc[RNG.integers(0, len(b), len(b))].T.values)[0, 1] for _ in range(nb)]
    return {"r": float(r0), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_studies": int(len(b))}


out = {"all": corr_ci(B)}
for k in ["forced", "open", "near"]:
    if (B.balance == k).sum() >= 10:
        out[k] = corr_ci(B[B.balance == k])
# open balance AND no aq/gas reported -> no closure can be imposed through the data at all
nb = B[~B.has_aq_gas]
if len(nb.dropna(subset=["Y_oil", "Y_char"])) >= 10:
    out["oil_char_only_studies"] = corr_ci(nb)
# closure null from observed phase-effect variances (studies with all four effects)
B4 = B.dropna(subset=PH)
V = np.diag(B4[PH].var().values)
P = np.eye(4) - np.ones((4, 4)) / 4
Cn = P @ V @ P
out["closure_null_projection"] = {"r_oil_char": float(Cn[0, 1] / np.sqrt(Cn[0, 0] * Cn[1, 1])),
                                  "phase_effect_sd": dict(zip(PH, np.sqrt(np.diag(V)).round(2).tolist())),
                                  "n_studies_all4": int(len(B4))}
out["empirical_corr_matrix_all4"] = B4[PH].corr().round(3).values.tolist()
perm = [np.corrcoef(B.dropna(subset=["Y_oil", "Y_char"]).Y_oil, RNG.permutation(B.dropna(subset=["Y_oil", "Y_char"]).Y_char))[0, 1] for _ in range(2000)]
out["permutation_null_95"] = [float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))]
out["balance_counts"] = B.balance.value_counts().to_dict()
B.to_csv(RES / "15_study_effects_by_phase.csv", index=False)
json.dump(out, open(RES / "15_closure_null.json", "w"), indent=1)
print(json.dumps(out, indent=1))
