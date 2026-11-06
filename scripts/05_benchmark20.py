"""20-algorithm benchmark + nested stacking, study-grouped vs random 5-fold CV.
Preprocessing (median imputation, scaling) is fitted inside each training fold.
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
import xgboost as xgb, lightgbm as lgb, catboost as cb
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from common import *


def P(est, scale=True):
    steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else []) + [est]
    return make_pipeline(*steps)


MODELS = {
    "Ridge": P(Ridge(alpha=3.0)),
    "Lasso": P(Lasso(alpha=0.05, max_iter=20000)),
    "ElasticNet": P(ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000)),
    "BayesRidge": P(BayesianRidge()),
    "Huber": P(HuberRegressor(alpha=1e-3, max_iter=2000)),
    "PLS": P(PLSRegression(n_components=12)),
    "KNN": P(KNeighborsRegressor(n_neighbors=7, weights="distance")),
    "SVR": P(SVR(C=30, epsilon=1.0, gamma="scale")),
    "KRR": P(KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01)),
    "GPR": P(GaussianProcessRegressor(ConstantKernel(100.0) * RBF(5.0) + WhiteKernel(30.0), normalize_y=True,
                                      n_restarts_optimizer=0, random_state=SEED)),
    "DT": P(DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=SEED), False),
    "RF": P(RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5, n_jobs=2, random_state=SEED), False),
    "ExtraTrees": P(ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.6, n_jobs=2, random_state=SEED), False),
    "Bagging": P(BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3), n_estimators=200, n_jobs=2, random_state=SEED), False),
    "AdaBoost": P(AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300, learning_rate=0.05, random_state=SEED), False),
    "GBDT": P(GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=SEED), False),
    "HistGB": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED),
    "XGBoost": xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                min_child_weight=3, reg_lambda=1.0, n_jobs=2, random_state=SEED),
    "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2, random_state=SEED),
    "CatBoost": cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0,
                                     thread_count=2, random_seed=SEED),
    "MLP": P(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=800,
                          early_stopping=True, random_state=SEED)),
}


def cv_model(name, X, y, folds):
    oof = np.zeros(len(y)); tr_r2 = []
    for tr, te in folds:
        m = clone(MODELS[name]).fit(X[tr], y[tr])
        oof[te] = np.ravel(m.predict(X[te]))
        tr_r2.append(metrics(y[tr], np.ravel(m.predict(X[tr])))["R2"])
    return name, oof, float(np.mean(tr_r2))


def nested_stack(X, y, d, base, outer):
    oof = np.zeros(len(y))
    for tr, te in outer:
        dtr = d.iloc[tr].reset_index(drop=True)
        inner = splits(dtr, "group", seed=SEED + 1)
        Z = np.zeros((len(tr), len(base))); Zt = np.zeros((len(te), len(base)))
        for j, nm in enumerate(base):
            for itr, ite in inner:
                Z[ite, j] = np.ravel(clone(MODELS[nm]).fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite]))
            Zt[:, j] = np.ravel(clone(MODELS[nm]).fit(X[tr], y[tr]).predict(X[te]))
        meta = LinearRegression(positive=True).fit(Z, y[tr])
        oof[te] = meta.predict(Zt)
    return oof


def run(tgt):
    d = load(); d = d[d[tgt].notna()].reset_index(drop=True)
    X = design(d).values; y = d[tgt].values
    rows = []
    preds = {}
    for kind in ["group", "random"]:
        folds = splits(d, kind)
        out = Parallel(n_jobs=10)(delayed(cv_model)(nm, X, y, folds) for nm in MODELS)
        for nm, oof, trr2 in out:
            r = metrics(y, oof); r.update(model=nm, split=kind, train_R2=trr2, gap=trr2 - r["R2"], target=tgt)
            # per-study R2 distribution (group split only)
            if kind == "group":
                preds[nm] = oof
            rows.append(r)
        print(tgt, kind, "done", flush=True)
    R = pd.DataFrame(rows)
    top = R[R.split == "group"].sort_values("RMSE").model.head(6).tolist()
    folds = splits(d, "group")
    st = nested_stack(X, y, d, top, folds)
    r = metrics(y, st); r.update(model="Stacking(top6)", split="group", target=tgt, train_R2=np.nan, gap=np.nan, base=",".join(top))
    rows.append(r); preds["Stacking(top6)"] = st
    R = pd.DataFrame(rows)
    pd.DataFrame(preds).assign(y=y, study_id=d.study_id.values).to_csv(RES / f"05_oof_group_{tgt}.csv", index=False)
    return R


if __name__ == "__main__":
    tg = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Y_oil", "Y_char"]
    allR = pd.concat([run(t) for t in tg])
    allR.to_csv(RES / "05_benchmark20.csv", index=False)
    print(allR.sort_values(["target", "split", "RMSE"]).to_string())
