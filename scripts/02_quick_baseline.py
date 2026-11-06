"""Quick probe: random vs study-grouped CV for each product phase; with/without protocol covariates."""
import sys, json, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

df = load()
rows = []
for tgt in ["Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "HHV_oil"]:
    d = df[df[tgt].notna()].reset_index(drop=True)
    for feat_name, cats in [("chem+process", CAT_CHEM), ("+protocol", CAT)]:
        X = design(d, cats=cats).values; y = d[tgt].values
        for kind in ["random", "group"]:
            oof = np.zeros(len(d))
            for tr, te in splits(d, kind):
                m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                      verbose=-1, random_state=SEED, n_jobs=6)
                m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
            r = metrics(y, oof); r.update(target=tgt, n=len(d), studies=d.study_id.nunique(), features=feat_name, split=kind)
            rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(RES / "02_quick_baseline.csv", index=False)
