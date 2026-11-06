"""Prepare an analysis table from RADAR V1.0.0 master_dataset.csv.

Read-only on the source; writes 10_ML分析/data/analysis_table.parquet (+csv)
and a data-audit json. All feature engineering here is row-wise (no
cross-row statistics), so it cannot leak across CV folds.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "master_dataset.csv"
OUT = ROOT / "data"

df0 = pd.read_csv(SRC, low_memory=False)
audit = {"raw_shape": list(df0.shape), "exact_duplicate_rows": int(df0.duplicated().sum())}
df = df0.drop_duplicates().reset_index(drop=True)

num = lambda c: pd.to_numeric(df[c], errors="coerce")

# ---------------- group id (independent unit = source study) ----------------
doi = df["DOI"].astype(str).str.strip().str.lower()
doi = doi.str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
df["study_id"] = pd.factorize(doi)[0]

# ---------------- categorical harmonisation ----------------
def solvent_class(s: str) -> str:
    s = str(s).lower()
    org = any(k in s for k in ["ethanol", "methanol", "acetone", "butanol", "propanol", "glycerol",
                                "thf", "dioxane", "tetralin", "phenol", "ethylene glycol", "isopropanol",
                                "formic", "acetic", "toluene", "cresol", "solvent"])
    water = any(k in s for k in ["water", "h2o", "aqueous", "deioni", "distilled"])
    if water and org:
        return "water_organic"
    if org:
        if "ethanol" in s: return "ethanol"
        if "methanol" in s: return "methanol"
        if "acetone" in s: return "acetone"
        return "other_organic"
    if water:
        return "water"
    return "other"

def catalyst_class(s: str) -> str:
    s0 = str(s)
    s = s0.lower().replace(" ", "")
    if s in ("none", "nan", "", "h2o", "no", "non-catalytic"):
        return "none"
    if any(k in s for k in ["k2co3", "na2co3", "kco3", "nahco3", "khco3", "carbonate", "cs2co3"]):
        return "alkali_carbonate"
    if any(k in s for k in ["naoh", "koh", "lioh", "ca(oh)2", "hydroxide", "mg(oh)2", "ba(oh)2"]):
        return "alkali_hydroxide"
    if any(k in s for k in ["formic", "hcooh", "nahcoo", "formate", "h2donor", "tetralin"]):
        return "h_donor_formate"
    if any(k in s for k in ["h2so4", "hcl", "acetic", "h3po4", "acid", "zncl2", "fecl3", "alcl3"]):
        return "acid_lewis"
    if any(k in s for k in ["zsm", "zeolite", "hzsm", "y-zeolite", "sapo", "mcm", "sba"]):
        return "zeolite"
    if any(k in s for k in ["ni", "co", "mo", "fe", "cu", "ru", "pt", "pd", "ce", "zr", "la", "mn", "zn"]):
        return "metal_oxide_supported"
    return "other"

def atmosphere_class(a, p) -> str:
    t = (str(a) + " " + str(p)).lower()
    if re.search(r"\bh2\b|hydrogen|h₂", t) and "n2" not in t.split("h2")[0][-3:]:
        if re.search(r"h2\s*\(|h2 pre|h2 [0-9]|h2,|^h2|h2 \(|hydrogen|h₂|\bh2\b", t):
            return "reducing_H2"
    if any(k in t for k in ["air", "o2", "oxid"]):
        return "oxidative_air"
    if any(k in t for k in ["n2", "n₂", "argon", " ar ", "ar ", "inert", "helium", "he ", "co2"]):
        return "inert_purge"
    if any(k in t for k in ["autogen", "autogenous", "sealed", "vacuum", "evacuated"]):
        return "autogenic"
    return "unreported"

def reactor_class(r) -> str:
    r = str(r).lower()
    if any(k in r for k in ["continuous", "plug", "flow", "pfr", "cstr", "pilot"]):
        return "continuous"
    if any(k in r for k in ["micro", "tubular", "bomb", "capillary", "10 ml", "20 ml", "sand bath"]):
        return "micro_batch"
    if r in ("nan", ""):
        return "unreported"
    return "batch_autoclave"

def extraction_class(s) -> str:
    s = str(s).lower()
    if s in ("nan", ""):
        return "unreported"
    if "dcm" in s or "dichloromethane" in s or "ch2cl2" in s: return "DCM"
    if "acetone" in s: return "acetone"
    if "ethyl acetate" in s or " ea " in s or "ea extraction" in s: return "ethyl_acetate"
    if "ether" in s: return "ether"
    if "chloroform" in s: return "chloroform"
    if "thf" in s or "tetrahydrofuran" in s: return "THF"
    if "toluene" in s or "hexane" in s or "benzene" in s: return "nonpolar"
    if "ethanol" in s or "methanol" in s: return "alcohol"
    return "filtration_or_other"

def basis_class(s) -> str:
    s = str(s).lower()
    if "ash-free" in s or "daf" in s or "ash free" in s: return "daf"
    if "carbon" in s: return "carbon_basis"
    if "dry" in s or "db" in s: return "dry"
    return "other_or_unreported"

df["solvent_class"] = df["solvent_or_medium"].map(solvent_class)
df["catalyst_class"] = df["catalyst"].map(catalyst_class)
df["atmosphere_class"] = [atmosphere_class(a, p) for a, p in zip(df["atmosphere"], df["pressure_reaction_MPa"])]
df["reactor_class"] = df["reactor"].map(reactor_class)
df["extraction_class"] = df["separation_method"].map(extraction_class)
df["basis_class"] = df["yield_basis"].map(basis_class)

# ---------------- numeric features (deployment-time available) ----------------
feat_num = {
    "C": "C_feed_wt_pct", "H": "H_feed_wt_pct", "O": "O_feed_wt_pct", "N": "N_feed_wt_pct",
    "S": "S_feed_wt_pct", "Ash": "Ash_feed_wt_pct", "Lignin": "Lignin_feed_wt_pct",
    "Cellulose": "Cellulose_feed_wt_pct", "Hemicellulose": "Hemicellulose_feed_wt_pct",
    "Extractives": "Extractives_feed_wt_pct", "HHV_feed": "HHV_feed_MJ_per_kg",
    "OC_feed": "O_C_feed_molar", "HC_feed": "H_C_feed_molar",
    "T": "T_reaction_C", "t": "t_residence_min", "SL": "IC_feed_wt_pct_slurry",
    "SBR": "water_biomass_ratio_kg_kg", "CatR": "cat_biomass_ratio_kg_kg",
    "HR": "heating_rate_C_per_min",
}
for k, c in feat_num.items():
    df[k] = num(c)
df["log_t"] = np.log1p(df["t"])
# hydrothermal severity factor (Overend-Chornet); T in C, t in min
df["log_R0"] = np.log10(np.clip(df["t"], 0.1, None) * np.exp((df["T"] - 100.0) / 14.75))
df["Lignin_imp"] = df["Lignin_imputed"].astype(str).str.lower().eq("true").astype(int)
df["Cell_imp"] = df["cellulose_imputed"].astype(str).str.lower().eq("true").astype(int)
df["year_num"] = num("year")

# ---------------- targets ----------------
tg = {"Y_oil": "Yield_biooil_wt_pct", "Y_char": "Yield_char_wt_pct",
      "Y_aq": "Yield_aqueous_wt_pct", "Y_gas": "Yield_gas_wt_pct", "Y_gaswater": "Yield_gas_water_wt_pct",
      "HHV_oil": "HHV_biooil_MJ_per_kg", "HHV_char": "HHV_char_MJ_per_kg",
      "C_oil": "C_biooil_wt_pct", "C_char": "C_char_wt_pct"}
for k, c in tg.items():
    df[k] = num(c)
# recomputed recoveries (source columns are mis-scaled, see audit)
df["ER_oil"] = df["Y_oil"] * df["HHV_oil"] / df["HHV_feed"]          # % of feed energy
df["ER_char"] = df["Y_char"] * df["HHV_char"] / df["HHV_feed"]
df["CR_oil"] = df["Y_oil"] * df["C_oil"] / df["C"]                     # % of feed carbon
df["CR_char"] = df["Y_char"] * df["C_char"] / df["C"]

e = num("Energy_yield_biooil_pct"); m = e.notna() & df["Y_oil"].notna() & df["HHV_oil"].notna()
audit["energy_yield_biooil_col_equals_Y_times_HHV_over_100"] = {
    "n": int(m.sum()),
    "median_ratio": float(np.median(e[m] / (df.loc[m, "Y_oil"] * df.loc[m, "HHV_oil"] / 100))),
    "note": "Source column labelled % of feed energy is numerically Y_oil*HHV_oil/100 (MJ per 100 g feed); true energy recovery recomputed as ER_oil = Y_oil*HHV_oil/HHV_feed."}
cy = num("Carbon_yield_biooil_pct"); m = cy.notna() & df["Y_oil"].notna() & df["C_oil"].notna()
audit["carbon_yield_biooil_col_ratio_to_true_recovery_fraction"] = {
    "n": int(m.sum()),
    "median_ratio": float(np.median(cy[m] / (df.loc[m, "Y_oil"] * df.loc[m, "C_oil"] / df.loc[m, "C"] / 100))),
    "note": "Source carbon-yield column equals Y*C_oil/1e4 (feed-carbon divisor omitted, ratio ~ C_feed/100); recomputed CR_oil."}

# physical sanity masks (do not edit values; flag)
for k in ["Y_oil", "Y_char", "Y_aq", "Y_gas"]:
    df.loc[(df[k] < 0) | (df[k] > 100), k] = np.nan
df["sum_oil_char"] = df[["Y_oil", "Y_char"]].sum(axis=1, min_count=2)
audit["oil_plus_char_gt_100"] = int((df["sum_oil_char"] > 100.5).sum())
df.loc[df["ER_oil"] > 110, "ER_oil"] = np.nan
df.loc[df["CR_oil"] > 110, "CR_oil"] = np.nan

audit["after_dedup_rows"] = int(len(df))
audit["n_studies"] = int(df["study_id"].nunique())
audit["targets_nonnull"] = {k: int(df[k].notna().sum()) for k in ["Y_oil", "Y_char", "Y_aq", "Y_gas", "ER_oil", "CR_oil", "HHV_oil", "ER_char", "HHV_char"]}
audit["category_counts"] = {c: df[c].value_counts().to_dict() for c in
                            ["process_subtype", "solvent_class", "catalyst_class", "atmosphere_class",
                             "reactor_class", "extraction_class", "basis_class", "Family_std"]}
pattern = df[["Y_oil", "Y_char", "Y_aq", "Y_gas"]].notna().astype(int).astype(str).agg("".join, axis=1)
audit["yield_observation_patterns(oil,char,aq,gas)"] = pattern.value_counts().to_dict()

keep = (["study_id", "DOI", "paper_title", "year_num", "Feedstock", "Family_std", "process_subtype",
         "solvent_class", "catalyst_class", "atmosphere_class", "reactor_class", "extraction_class",
         "basis_class", "catalyst", "solvent_or_medium"] + list(feat_num) +
        ["log_t", "log_R0", "Lignin_imp", "Cell_imp"] + list(tg) + ["ER_oil", "ER_char", "CR_oil", "CR_char"])
out = df[keep].copy()
out.to_csv(OUT / "analysis_table.csv", index=False, encoding="utf-8-sig")
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "data_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps({k: v for k, v in audit.items() if k != "category_counts"}, indent=1, default=str))
for c, v in audit["category_counts"].items():
    print(c, v)
