"""Target-specific boosting-round check for the GPBoost baseline (fairness of 14_gpboost_baseline.py).

Fold-0 outer-training studies, study-grouped inner folds (same as 14), GPB-RE model, gpb.cv with early stopping
(patience 50, cap 3000) for learning rates 0.1 and 0.03. Validation studies are unseen, so the validation loss
is the zero-shot (random effect = 0) error, the standard GPBoost tuning route. Also refits GPB-RE on all fold-0
training studies with the selected rounds and reports the learned covariance (error variance, study variance) to
see whether the study variance collapses to 0 with fewer rounds.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import gpboost as gpb
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *

BASE = dict(num_leaves=31, min_data_in_leaf=10, feature_fraction=0.8, lambda_l2=1.0, verbose=-1, num_threads=8, seed=SEED)
tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_aq", "Y_gas", "ER_oil", "HHV_oil"]
out = {}
for tgt in tg:
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values.astype(float); y = d[tgt].values
    gi = pd.factorize(d.study_id.values)[0].astype(int)
    tr, _ = splits(d, "group")[0]
    inner = splits(d.iloc[tr].reset_index(drop=True), "group", seed=SEED + 3)
    res = {}
    for lr in [0.1, 0.03]:
        gm = gpb.GPModel(group_data=gi[tr], likelihood="gaussian")
        cv = gpb.cv(params={**BASE, "learning_rate": lr}, train_set=gpb.Dataset(X[tr], y[tr]), gp_model=gm,
                    folds=inner, num_boost_round=3000, early_stopping_rounds=50, metric="mse",
                    use_gp_model_for_validation=True, seed=SEED)
        key = [k for k in cv if k.endswith("-mean")][0]
        best = len(cv[key]); loss = float(cv[key][-1])
        gm2 = gpb.GPModel(group_data=gi[tr], likelihood="gaussian")
        gpb.train(params={**BASE, "learning_rate": lr}, train_set=gpb.Dataset(X[tr], y[tr]), gp_model=gm2, num_boost_round=best)
        cp = np.asarray(gm2.get_cov_pars()).ravel().tolist()
        res[str(lr)] = {"best_rounds": best, "cv_mse_zero_shot": loss, "cov_pars_err_study": cp,
                        "var_y_train": float(np.var(y[tr]))}
        print(tgt, lr, res[str(lr)], flush=True)
    out[tgt] = res
    json.dump(out, open(RES / f"14b_gpboost_rounds_{'_'.join(tg)}.json", "w"), indent=1)
