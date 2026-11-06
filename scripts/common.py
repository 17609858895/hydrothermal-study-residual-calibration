"""Shared feature definitions, splits and metrics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "analysis_table.csv"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 2026

NUM_FEED = ["C", "H", "O", "N", "S", "Ash", "Lignin", "Cellulose", "Hemicellulose", "Extractives",
            "HHV_feed", "OC_feed", "HC_feed", "Lignin_imp"]
NUM_PROC = ["T", "log_t", "log_R0", "SL", "SBR", "CatR", "HR"]
CAT_CHEM = ["process_subtype", "solvent_class", "catalyst_class", "atmosphere_class", "Family_std"]
CAT_PROTO = ["reactor_class", "extraction_class", "basis_class"]   # measurement / work-up protocol
CAT = CAT_CHEM + CAT_PROTO
PHASES = ["Y_oil", "Y_char", "Y_aq", "Y_gas"]


def load(subset: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df["process_subtype"] != "APR"].reset_index(drop=True)
    if subset == "HTL":
        df = df[df["process_subtype"].isin(["HTL", "HTL_solvothermal"])].reset_index(drop=True)
    return df


def design(df: pd.DataFrame, cats=CAT, nums=None, categories: dict | None = None):
    """One-hot design matrix with fixed category levels (levels are label-free metadata)."""
    nums = nums or (NUM_FEED + NUM_PROC)
    X = df[nums].astype(float).copy()
    for c in cats:
        levels = categories[c] if categories else sorted(df[c].astype(str).unique())
        for lv in levels:
            X[f"{c}={lv}"] = (df[c].astype(str) == lv).astype(float)
    return X


def splits(df, kind="group", n_splits=5, seed=SEED):
    idx = np.arange(len(df))
    if kind == "random":
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(idx))
    # shuffled GroupKFold by study (sklearn 1.5 GroupKFold is deterministic; permute group ids)
    rng = np.random.default_rng(seed)
    g = df["study_id"].values
    ug = np.unique(g)
    perm = dict(zip(ug, rng.permutation(len(ug))))
    gp = np.array([perm[x] for x in g])
    return list(GroupKFold(n_splits).split(idx, groups=gp))


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": mean_absolute_error(y, p)}
