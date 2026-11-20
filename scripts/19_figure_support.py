"""Supporting analyses for the revised figure system (no change to any reported model).

A. Nested literature structure in predictor space
   - Records are represented by the same composition/process/protocol design used by the predictors
     (numeric: median-imputed and standardized; categories: one-hot).
   - nearest-neighbour diagnostic: compare distances to other records in the same or a different
     source study; for random-row partitions, identify the study of each test record's nearest
     training record.
   - two-dimensional t-SNE map for display only (no analysis uses its coordinates).
Outputs: results/19_nn_leakage.json, results/19_nn_distances.csv, results/19_tsne_records.csv
"""
import argparse
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import *


def standardized_design(d):
    X = design(d)
    num = [c for c in NUM_FEED + NUM_PROC if c in X.columns]
    Z = X.copy()
    Z[num] = Z[num].fillna(Z[num].median())
    Z[num] = (Z[num] - Z[num].mean()) / (Z[num].std() + 1e-9)
    return Z.values.astype(float)


def same_and_other_distances(Z, studies):
    same = np.full(len(Z), np.inf)
    other = np.empty(len(Z))
    for study in np.unique(studies):
        in_study = studies == study
        within = cdist(Z[in_study], Z[in_study])
        np.fill_diagonal(within, np.inf)
        same[in_study] = within.min(axis=1)
        other[in_study] = cdist(Z[in_study], Z[~in_study]).min(axis=1)
    return same, other


def nn_leakage(d, label):
    Z = standardized_design(d); g = d.study_id.values
    nn = NearestNeighbors(n_neighbors=1).fit(Z)
    ind = nn.kneighbors(return_distance=False)[:, 0]
    same_any = float(np.mean(g[ind] == g))
    rand = []
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Z):
        m = NearestNeighbors(n_neighbors=1).fit(Z[tr])
        _, j = m.kneighbors(Z[te])
        rand.append(g[tr][j[:, 0]] == g[te])
    rand = np.concatenate(rand)
    # distance to nearest same-study vs nearest other-study record
    ds, do = same_and_other_distances(Z, g)
    comparable = np.isfinite(ds)
    ties = np.isclose(ds, do, rtol=0, atol=1e-9)
    strictly_closer = (ds < do) & ~ties
    if label == "all records":
        pd.DataFrame({"nn_same_study": ds[comparable], "nn_other_study": do[comparable]}).to_csv(
            RES / "19_nn_distances.csv", index=False
        )
    return {"subset": label, "records": int(len(d)), "studies": int(len(np.unique(g))),
            "nearest_record_same_study": same_any,
            "random_split_test_nearest_train_same_study": float(rand.mean()),
            "median_dist_same_study": float(np.median(ds[comparable])),
            "median_dist_other_study": float(np.median(do[comparable])),
            "share_same_study_closer": float(np.mean(strictly_closer[comparable])),
            "share_same_study_strictly_closer_all_records": float(np.mean(strictly_closer)),
            "same_and_other_distance_ties": int(ties.sum()),
            "records_with_same_study_neighbour": int(comparable.sum()),
            "identical_predictors_same_study": int(np.count_nonzero(ds == 0)),
            "identical_predictors_other_study": int(np.count_nonzero(do == 0))}


def part_a(neighbours_only=False):
    d = load().reset_index(drop=True)
    out = {"all": nn_leakage(d, "all records")}
    for t in ["Y_oil", "Y_char"]:
        out[t] = nn_leakage(d[d[t].notna()].reset_index(drop=True), t)
    with (RES / "19_nn_leakage.json").open("w", encoding="utf-8") as output:
        json.dump(out, output, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    if neighbours_only:
        return
    Z = standardized_design(d)
    emb = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=SEED).fit_transform(Z)
    n = d.groupby("study_id").size()
    pd.DataFrame({"study_id": d.study_id, "x": emb[:, 0], "y": emb[:, 1], "process_subtype": d.process_subtype,
                  "family": d.Family_std, "year": d.year_num, "study_runs": d.study_id.map(n).values,
                  "has_oil": d.Y_oil.notna(), "has_char": d.Y_char.notna()}).to_csv(RES / "19_tsne_records.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize study nesting in predictor space.")
    parser.add_argument("--neighbours-only", action="store_true", help="Update distances without recalculating t-SNE.")
    part_a(parser.parse_args().neighbours_only)
