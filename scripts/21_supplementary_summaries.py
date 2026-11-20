"""Summarize study coverage and the isolated first-fold mixed-effect comparison."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RES


def primary_study_summaries():
    coverage_rows = []
    comparison_rows = []
    for target in ("Y_oil", "Y_char"):
        episodes = pd.read_csv(RES / f"09_srgp_raw_{target}.csv")
        episodes = episodes[episodes.model.isin(["SR-GP", "offset"])]
        for budget, group in episodes.groupby("k"):
            study_error = group.groupby(["study", "model"]).rmse_study.mean().unstack()
            comparison_rows.append({
                "target": target, "k": budget, "studies": len(study_error),
                "fraction_studies_srgp_lower_rmse": (study_error["SR-GP"] < study_error["offset"]).mean(),
            })
            for model, method in group.groupby("model"):
                weighted = method.assign(covered=method.cover * method.n)
                counts = weighted.groupby("study")[["covered", "n"]].sum()
                coverage = counts.covered / counts.n
                coverage_rows.append({
                    "target": target, "k": budget, "model": model, "studies": len(coverage),
                    "median_study_coverage": coverage.median(),
                    "fraction_studies_coverage_below_0_8": (coverage < 0.8).mean(),
                })
    pd.DataFrame(coverage_rows).to_csv(RES / "21_study_coverage.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(RES / "21_matched_study_errors.csv", index=False)


def first_fold_sensitivity():
    generator = np.random.default_rng(2044)
    rows = []
    for target in ("Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "HHV_oil"):
        suffix = "" if target == "Y_oil" else "_tuned"
        episodes = pd.read_csv(RES / f"14_gpboost_raw_{target}{suffix}.csv")
        for budget, group in episodes[episodes.fold == 0].groupby("k"):
            sums = group.groupby(["study", "model"])[["sse", "sst"]].sum().unstack()
            errors = sums["sse"][["SR-GP", "GPB-RE", "GPB-RE+GP"]].to_numpy()
            reference = sums["sst"]["SR-GP"].to_numpy()
            scores = 1 - errors.sum(axis=0) / reference.sum()
            sampled = generator.integers(0, len(sums), size=(2000, len(sums)))
            bootstrap_errors = errors[sampled].sum(axis=1)
            differences = (bootstrap_errors[:, 1:].min(axis=1) - bootstrap_errors[:, 0]) / reference[sampled].sum(axis=1)
            lower, upper = np.quantile(differences, [0.025, 0.975])
            rows.append({
                "target": target, "k": budget, "studies": len(sums),
                "Q2_SRGP": scores[0], "Q2_GPB_RE": scores[1], "Q2_GPB_RE_GP": scores[2],
                "difference_vs_better_GPB": scores[0] - scores[1:].max(),
                "lower_95": lower, "upper_95": upper,
            })
    pd.DataFrame(rows).to_csv(RES / "21_first_fold_gpboost.csv", index=False)


if __name__ == "__main__":
    primary_study_summaries()
    first_fold_sensitivity()
