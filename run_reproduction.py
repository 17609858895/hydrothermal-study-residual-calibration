"""Execute the study analysis in dependency order, with optional sensitivities."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
TARGETS = "Y_oil,Y_char,Y_aq,Y_gas,ER_oil,HHV_oil"


def run(script: str, *args: str, base: str = "lightgbm", dry_run: bool = False) -> None:
    command = [sys.executable, str(SCRIPTS / script), *args]
    print("Running:", " ".join(["python", f"scripts/{script}", *args]), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", SRGP_BASE=base, GPB_ROUNDS="1500", GPB_DRAWS="20")
    logs = ROOT / "results" / "execution_logs"
    logs.mkdir(parents=True, exist_ok=True)
    suffix = "_" + base + ("_" + args[0].replace(",", "_") if args else "")
    with (logs / f"{Path(script).stem}{suffix}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def primary_summary_alias(dry_run: bool = False) -> None:
    source = ROOT / "results" / "09_srgp_summary_Y_oil_Y_char.csv"
    destination = ROOT / "results" / "09_srgp_summary_oil_char.csv"
    print("Creating the primary-response summary alias", flush=True)
    if dry_run:
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copyfile(source, destination)
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
        raise RuntimeError("Primary-response summary alias verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, help="Manually downloaded official RADAR Version 1.0.0 TAR")
    parser.add_argument("--profile", choices=["core", "full"], default="core",
                        help="full adds all six-response sensitivities and the publication figures")
    parser.add_argument("--secondary", action="store_true", help="Run the four additional LightGBM responses")
    parser.add_argument("--catboost", action="store_true", help="Run CatBoost residual correction for all six responses")
    parser.add_argument("--figures", action="store_true", help="Generate figures; includes their sensitivity dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without reading data or training")
    args = parser.parse_args()
    full = args.profile == "full" or args.figures
    if args.tar:
        print(f"Verifying source data from {args.tar}", flush=True)
        if not args.dry_run:
            from extract_radar_data import extract_verified_csv
            extract_verified_csv(args.tar, ROOT / "data" / "master_dataset.csv")
    elif not args.dry_run:
        from extract_radar_data import digest, EXPECTED_SHA256
        data = ROOT / "data" / "master_dataset.csv"
        if not data.is_file() or digest(data) != EXPECTED_SHA256:
            parser.error("Provide --tar with the official RADAR archive, or verify data/master_dataset.csv first")
    for script in ["01_prepare_data.py", "02_quick_baseline.py", "03_study_effect_contrastive.py",
                   "04_fe_dml.py", "05_benchmark20.py", "06_study_conformal.py",
                   "07_within_study_diagnostic.py", "08_meta_anchor.py"]:
        run(script, dry_run=args.dry_run)
    run("09_srgp.py", "Y_oil,Y_char", dry_run=args.dry_run)
    primary_summary_alias(args.dry_run)
    for script in ["10_lab_effect_explain.py", "11_active_anchor.py", "13_mo_srgp.py",
                   "14_gpboost_baseline.py", "15_closure_null.py"]:
        run(script, dry_run=args.dry_run)
    if full or args.secondary:
        run("09_srgp.py", "Y_aq,Y_gas,ER_oil,HHV_oil", dry_run=args.dry_run)
    if full or args.catboost:
        run("09_srgp.py", TARGETS, base="catboost", dry_run=args.dry_run)
    if full:
        run("16_publication_figures.py", dry_run=args.dry_run)
    print("Commands listed." if args.dry_run else "Requested analysis stages completed.")


if __name__ == "__main__":
    main()
