# Condition-dependent calibration of hydrothermal biomass models

Analysis code for **Condition-dependent calibration of literature-trained hydrothermal yield predictions with a few local measurements**.

The analysis retains 3,548 hydrothermal conversion records from 186 DOI-based source-study groups after removal of 136 exact duplicate rows and nine aqueous-phase-reforming records. A source-study identifier does not identify an independent laboratory. The primary outcomes are reported biocrude and solid/char yields; supplementary analyses cover aqueous and gas yields, biocrude heating value, and a calculated oil energy ratio.

## Data

Download the official **RADAR Version 1.0.0** TAR manually from [the dataset page](https://doi.org/10.22000/8bnbhqccf4aq7kea), following its terms and attribution requirements. The dataset is attributed to Seifallah Elfetni and is distributed under CC BY 4.0. Its original papers retain their respective rights. This repository does not redistribute the TAR, experimental CSV or source-publication PDFs.

`extract_radar_data.py` reads only the regular member ending in `/master_dataset.csv`, streams it to the fixed local data path and verifies the official CSV SHA-256:

```
c8827a378879ce9d8d1ed7539e451b7b741eec30c5cc070487e3f467a4a42a6b
```

It does not extract arbitrary archive members. A different CSV is rejected rather than overwritten. The analysis uses this specific release; a similarly named mirror or newer release should not be silently substituted.

## Environment and quick start

Python 3.12.3 was used. The installed package versions were read from environment metadata and are recorded in `environment_versions.json` and `requirements.txt`. The multi-output model uses the CPU build of PyTorch. Create an isolated environment and install the recorded dependencies:

```bash
python -m venv .venv
# Activate the environment using your operating system's usual command.
python -m pip install -r requirements.txt
python validate_package.py
```

Run the complete analysis and publication-figure dependencies from the manually downloaded TAR:

```bash
python run_reproduction.py --tar /path/to/official_radar_v1.tar --profile full
```

The core profile runs the primary-response analyses, broad benchmark, development comparators, active design, mixed-effect comparison and phase diagnostics. The full profile also runs all supplementary response/categorical-boosting sensitivities needed by the complete figure system:

```bash
python run_reproduction.py --tar /path/to/official_radar_v1.tar --profile core
python run_reproduction.py --profile core --secondary --catboost
python run_reproduction.py --profile full --dry-run
```

The second command reruns core stages before adding sensitivities; it is not a cached continuation. Use individual commands below for selective reruns. Full training includes grouped nested fits and is computationally intensive. Output logs are written under `results/execution_logs/`. The workflow stops if a stage fails. Generated figures and results stay inside the repository and are ignored by Git.

## Individual analysis stages

After `python extract_radar_data.py /path/to/official_radar_v1.tar`, run the following in order. The workflow above runs these commands with the recorded defaults.

```bash
python scripts/01_prepare_data.py
python scripts/02_quick_baseline.py
python scripts/03_study_effect_contrastive.py
python scripts/04_fe_dml.py
python scripts/05_benchmark20.py
python scripts/06_study_conformal.py
python scripts/07_within_study_diagnostic.py
python scripts/08_meta_anchor.py
python scripts/09_srgp.py Y_oil,Y_char
python -c "from run_reproduction import primary_summary_alias; primary_summary_alias()"
python scripts/10_lab_effect_explain.py
python scripts/11_active_anchor.py
python scripts/13_mo_srgp.py
python scripts/14_gpboost_baseline.py
python scripts/15_closure_null.py
```

GPBoost iterations are response specific (see below). The char comparison reported with biocrude uses 3,000 iterations; with PowerShell:

```powershell
$env:GPB_ROUNDS='3000'; $env:GPB_TAG='_tuned'
python scripts/14_gpboost_baseline.py Y_char
Remove-Item Env:GPB_ROUNDS, Env:GPB_TAG
python scripts/18_gpboost_gap_vs_transfer.py tuned
```

The controlled summary alias copies `09_srgp_summary_Y_oil_Y_char.csv` to the equivalent `09_srgp_summary_oil_char.csv` name expected by the figure entry point and verifies identical content.

The optional LightGBM secondary-response command is:

```bash
python scripts/09_srgp.py Y_aq,Y_gas,ER_oil,HHV_oil
```

The matching GPBoost comparisons run one response per process with `GPB_TAG=_tuned` and `GPB_ROUNDS` set to 566 (`Y_aq`), 849 (`Y_gas`), 3000 (`ER_oil`) or 381 (`HHV_oil`), followed by `python scripts/18_gpboost_gap_vs_transfer.py tuned`. The workflow runs these with `--secondary` or `--profile full`.

To reproduce the CatBoost sensitivity across all six responses, set `SRGP_BASE=catboost` for that command only. For example, in PowerShell:

```powershell
$env:SRGP_BASE='catboost'
python scripts/09_srgp.py Y_oil,Y_char,Y_aq,Y_gas,ER_oil,HHV_oil
Remove-Item Env:SRGP_BASE
```

The workflow's `--catboost` option supplies that environment value to its child process. The full profile then runs the corrected nearest-neighbour diagnostic, the extended panel analyses and the supplementary summaries before generating the five main figures, two supplementary figures and graphical abstract. These stages can also be run individually after their numerical dependencies exist.

```bash
python scripts/19_figure_support.py
python scripts/20_panel_support.py
python scripts/21_supplementary_summaries.py
python scripts/16_publication_figures.py
python scripts/17_graphical_abstract.py
```

The main figure exports contain 11, 10, 11, 8 and 10 independently rendered panels. The supplementary figures contain two panels and one panel. Each figure has a combined PNG/PDF and independent PNG/PDF panels under its `subfigures/` directory. The graphical abstract is generated as PNG, PDF and editable SVG. The common palette is `au_nimno_soft_8`, with ordinary-weight text. Rendering writes alignment and palette summaries to `results/figure_qa/`.

`19_figure_support.py` excludes the query record from its neighbours and uses direct Euclidean distances to distinguish exactly matching predictor vectors. It reports ties and studies with a single record. The strict within-study-neighbour fraction uses every record as its denominator. `--neighbours-only` updates these summaries without rerunning t-SNE. Numeric imputation and scaling for this descriptive diagnostic use the full display subset.

`20_panel_support.py` calculates corpus support, per-study validation errors, paired RMSE differences with study-bootstrap intervals, four-phase RMSE, interval-width strata, active-design summaries and three illustrative held-out studies. The illustrations are selected retrospectively near quartiles of the observed calibration-error ratio and are refitted with three variance-selected anchors. They illustrate the method on measured candidate sets. Individual parts can be selected by letter, for example `python scripts/20_panel_support.py a c d e f h`; part `b` reruns the pooled benchmark predictions and part `g` refits the three illustrations. The fixed sequence used for all panels is `a b c d e f g h`.

`21_supplementary_summaries.py` computes study-level coverage and error-comparison fractions and repeats the GPBoost comparison using only the first outer test fold. Its bootstrap resamples studies with all their matched episodes. The script produces three CSV summaries and includes all six responses and anchor budgets, with the first-fold five-anchor primary-yield results corresponding to the supplementary sensitivity comparison.

The repository reproduces numerical analyses and figure rendering. Word typesetting, reference-library integration and submission-form materials are maintained separately by the authors.

## Interpretation and evaluation

- **21 single-model benchmark:** the historical `05_benchmark20.py` filename is retained, but the script defines 21 prespecified regressors. Its additional stacked model selects members using complete-outcome grouped out-of-fold performance before outer-fold fitting; it is exploratory and is excluded from the principal 21-model figure.
- **Two score references:** ordinary benchmark R-squared uses the pooled evaluation mean. Few-shot prediction Q-squared uses the corresponding outer-training mean in the denominator, pooling query evaluations across repeated anchor sets. Repeated episodes are not independent publications.
- **Few-shot task:** anchors are measured responses from the otherwise held-out source study. They are used to predict its remaining recorded runs. This is retrospective study adaptation, distinct from zero-shot transfer or prospective experimental validation.
- **Intervals:** calibration pools dependent query/episode scores. Results are empirical held-out coverage and width, without an exact group-level finite-sample coverage claim.
- **Process contrasts:** fixed-effect double machine learning estimates within-study adjusted associations. A causal interpretation additionally requires exposure overlap and control of remaining within-study confounding. Heterogeneity displays are exploratory.
- **Yield bases and source processing:** reported yield-basis categories and archive-supplied calculated/imputed descriptors are retained. Prediction errors are in reported yield percentage points; the calculated energy ratio is a physical recovery measure only when component denominators are compatible.
- **Mixed-effect boosting comparison:** GPBoost iterations were selected for each response by `14b_gpboost_tune_rounds.py` using the study-grouped inner folds of the first outer training set, early stopping after 50 iterations and a 3,000-iteration cap. Selected values were 1,438 (biocrude, using the recorded 1,500-iteration run), 2,994 (char), 566 (aqueous), 849 (gas), 2,998 (energy ratio) and 381 (heating value), with values near the cap rounded to 3,000. These budgets were reused across outer folds. Studies in later test folds consequently contributed to iteration selection. The five-fold GPBoost comparison is exploratory, and `21_supplementary_summaries.py` provides the first-fold sensitivity whose test studies were excluded from that selection. With a uniform 1,500 iterations the heating-value study variance collapses to nearly zero. The full profile also reruns that sensitivity and writes `18_gpboost_gap_by_target_fixed.csv`. `18_gpboost_gap_vs_transfer.py tuned` combines the response-specific runs into `14_gpboost_summary_six_targets.csv` and reports study-bootstrap intervals from 2,000 resamples. Correlations across six responses are descriptive.
- **Anchor design:** all methods select different record indices without replacement, including when descriptors are duplicated. Variance-based and maximum-minimum-distance designs use at most 60 sampled candidates in large studies, while random anchors are sampled from the complete eligible study. The remaining recorded runs are queries.

In the active-design summary, `win_rate_vs_random` is the fraction of studies where IVR has lower mean query RMSE than random selection. That same IVR comparison is repeated across the strategy rows for a given target and anchor budget; it is not a separate win rate for each displayed strategy. In residual-correction summaries, `RMSE_study_median` is the median of study-by-anchor-draw RMSE values, whereas the active-design summary first averages random draws within a study.

## Package validation

`validate_package.py` compiles source text and tests unique maximum-minimum-distance anchor selection for identical conditions, repeated conditions and non-consecutive candidate indices. It also checks exact neighbour distances, cross-study ties and single-record studies. The package workflow can be inspected with `--dry-run`. Validation of this release covers compilation, these regressions, the full command sequence, supplementary numerical summaries and rendering against existing study results. A fresh-environment rerun of all model training remains an independent reproduction step.

Numerical results may depend on software builds and parallel linear algebra. Compare the statistical task, data version, sample support, metric reference and reported summaries when assessing reproduction. Keep the original RADAR attribution and cite the relevant primary publications when using the experimental compilation.
