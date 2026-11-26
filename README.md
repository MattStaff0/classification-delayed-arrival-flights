# Flight Delay Classification

> **AI-assistance notice:** This document was written with help from an AI
> assistant (Claude, via Claude Code). The research questions, the choices
> between approaches, and the conclusions are the team's. The assistant reformatted
> and edited the text, proposed some methods and checks (for example, the
> test-set freeze), and formatted the tables.
> Every number comes from a committed results file or W&B run named
> next to it. The team reviewed the content and is responsible for it. It is
> acknowledged here per the course's AI-use policy.

## At a glance

| | |
|---|---|
| **Task** | Binary classification: will a US domestic flight arrive 15+ min late? Booking-time features only (no weather, no live delays) |
| **Data** | 6.9M flights from BTS (Jul 2025 – Jun 2026), split by date: 5.1M train / 591K validation / 1.2M held-out test |
| **Final model** | XGBoost on 13 features, tuned with a 90-trial hyperopt search |
| **Test PR-AUC** | **0.368**, 1.56× the no-skill baseline (0.236) and above logistic regression (0.352) |
| **Test recall / precision** | Catches **63%** of late flights; flagged flights are late 35% of the time vs. a 24% base rate |
| **Test set** | Scored **once**, at the end, with hash-locked models and thresholds fixed on validation. Score vs. baseline fell only 4.7% from validation to test |

What was done:

- **Compared three models:** logistic regression baseline, XGBoost, and
  LightGBM. LightGBM tied XGBoost; XGBoost was picked because it trains
  3.6× faster.
- **Tested ideas and dropped the ones that didn't work:** a historical
  late-rate feature and 12 extra months of training data did not improve
  XGBoost on validation, so neither was adopted.
- **Analyzed the errors with SHAP:** scheduled time of day drives ~40% of
  predictions. The model ranks large hubs (ORD, JFK, CLT) and evening flights
  worst.
- **Made it reproducible:** locked dependencies (uv), every run tracked in
  Weights & Biases, and every reported number traced to a committed results
  file.

**Stack:** Python, pandas, scikit-learn, XGBoost, LightGBM, hyperopt, SHAP,
Weights & Biases, Parquet, uv.

Details and sources: [`docs/findings.md`](docs/findings.md); final scores:
`reports/final/eval4_metrics.json`.

Predicting whether a scheduled flight will arrive 15+ minutes late (`ArrDel15`),
using only information known at booking time. Data is BTS Reporting Carrier
On-Time Performance, July 2025 – June 2026.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and
dependencies (it replaced the earlier plain `venv` + `pip` setup). The Python
version is pinned in `.python-version` and exact package versions are locked
in `uv.lock`.

```bash
uv sync
```

This creates `.venv` with the pinned Python and the locked packages. Run
project commands with `uv run <command>`, or activate the environment with
`source .venv/bin/activate`. Add or remove packages with `uv add <pkg>` /
`uv remove <pkg>` rather than `pip install`.

**macOS only:** XGBoost needs the OpenMP runtime, which pip/uv packages do not
include. Without it, `import xgboost` fails with `libomp.dylib could not be
loaded`. Install it once per machine:

```bash
brew install libomp
```

`requirements.txt` is kept for environments without uv (e.g. Google Colab)
and is generated from the lock file:

```bash
uv export --no-hashes --format requirements-txt > requirements.txt
```

## Get the data

```bash
uv run scripts/fetch_bts_data.py --out data/raw --dry-run   # list months and check disk space
uv run scripts/fetch_bts_data.py --out data/raw             # 2025-07 .. 2026-06, ~3.5 GB
```

Downloads the BTS Reporting Carrier On-Time Performance monthly files from
TranStats, extracts each month's CSV, checks it holds the requested month, and
saves it as `data/raw/YYYY-MM-flight-data.csv`. Months already present are
skipped. Use `--start YYYY-MM --end YYYY-MM` for a different range (e.g.
`--start 2024-07` for EXP-2) and `--force` to re-download.

> **AI-generated code notice:** `scripts/fetch_bts_data.py` was written with
> an AI coding assistant (Claude). The team reviewed and ran it, and is
> responsible for it; it is acknowledged here per the course's AI-use policy.

## Build the processed data

```bash
uv run scripts/preprocess.py                      # base: data/raw -> data/processed
uv run scripts/preprocess.py --dataset fullyear   # EXP-2: data/raw + data/raw_2024 -> data/processed_fullyear
uv run scripts/preprocess.py --dataset aprjun2025 # EXP-2 run B: data/raw + Apr-Jun 2025 -> data/processed_aprjun2025
```

Run from the project root. Filters to completed flights, drops the 3 bad rows,
splits by date, and writes `{train,val,test}.parquet`. Stops with an error if
any check fails; re-running overwrites that dataset's files only. See
`docs/architecture/data-dictionary.md` (Processed Files) for the contents.

- **`--dataset`** picks a row of the `DATASETS` table at the top of
  `scripts/preprocess.py`: which raw files to read (folder + file pattern), where
  to write, and the file count and train row count the run must find. The
  default, `base`, is the project's dataset (MOD-1/2/3, EXP-4) and reproduces
  `data/processed` exactly.
- **`fullyear`** (EXP-2) adds Jul 2024 – Jun 2025 to train: 24 files, 12,053,138
  train rows. Download the extra months first:
  `uv run scripts/fetch_bts_data.py --start 2024-07 --end 2025-06 --out data/raw_2024`,
  then check them with `uv run python notebooks/EXP_2_Check_new_months.py`
  (read-only).
- **`aprjun2025`** (EXP-2 run B) adds only Apr–Jun 2025 from `data/raw_2024`
  (file pattern `2025-0[4-6]-flight-data.csv`): 15 files, 6,880,310 train rows.
- **Every dataset asserts the same validation and test sizes** (591,206 /
  1,199,540), so a dataset that changed them fails before writing anything. Each
  dataset has its own output folder, so a new one can never overwrite
  `data/processed`. To add a dataset (e.g. only Apr–Jun 2025 for EXP-2's
  follow-up), add a row with a file pattern such as `2025-0[4-6]-flight-data.csv`.

## Scripts

- `scripts/fetch_bts_data.py` — download and extract the monthly data (AI-generated, see notice above)
- `scripts/utils.py` — row counts and late rates per month
- `scripts/preprocess.py` — build the train/val/test Parquet files
- `scripts/export_wandb_runs.py` — save every W&B run with a given name and tag (e.g. all hyperopt trials) to a CSV in `reports/`, e.g. `uv run python scripts/export_wandb_runs.py --run MOD-2 --tag hyperopt-trial --out reports/xgboost/mod2_trials.csv`. Takes about 2 minutes

`docs/readme.html` is the BTS field documentation for all 109 columns.

## Running the baseline model

- Run from the project root
- `uv run python -m src.baseline.logistic_regression` - runs the logistic regression baseline, adds metrics to `reports/logistic_regression`

## Running XGBoost (MOD-2)

Run from the project root. On macOS, install `libomp` first (see Setup).

```bash
uv run python -m src.ensemble.xgboost_model             # one run with the default hyperparameters
uv run python -m src.ensemble.xgboost_model --tune 90   # hyperopt search: 90 trials, then retrain the best
uv run python -m src.ensemble.xgboost_model --params reports/xgboost/mod2_metrics.json   # retrain the final tuned model
```

- **Default run:** trains once with `DEFAULT_HYPERPARAMS_DICT` (XGBoost's
  defaults) and saves the outputs below. Takes about 15 seconds on an M-series Mac.
- **`--tune N`:** runs `N` hyperopt (TPE) trials over 7 hyperparameters
  (`learning_rate`, `max_depth`, `min_child_weight`, `subsample`,
  `colsample_bytree`, `reg_lambda`, `max_cat_threshold`), picking the
  configuration with the highest validation PR-AUC. Each trial is logged to W&B
  as a `MOD-2` run tagged `hyperopt-trial` and writes no files. The best
  configuration is then retrained and saved, tagged `final`. Trials average
  under a minute (longer at low learning rates), so 90 trials take about 1–2 hours.
- **Long runs:** keep the Mac awake and plugged in, lid open:
  ```bash
  caffeinate -i uv run python -m src.ensemble.xgboost_model --tune 90
  ```
  Python loads the code at start, so editing files during a run is safe, but
  do not switch branches: the final step writes its outputs into whatever
  branch is checked out when it finishes. Avoid training another model at the
  same time; the search uses every CPU core.
- **`--params PATH`:** see [Running a saved configuration](#running-a-saved-configuration---params---set).
- **Outputs** (default run, the `final` run after `--tune`, or a `--params`
  run; all overwrite the previous files):
  - `reports/xgboost/mod2_metrics.json`: run info, `train_seconds`, every hyperparameter used, `best_iteration`, `train` and `validation` blocks
  - `reports/xgboost/figures/mod2_pr_curve.png`, `mod2_thresholds.png`
  - `models/mod2_xgboost.joblib`: the fitted model and its validation threshold
- Prefix any command with `WANDB_MODE=disabled` to run without W&B.

## Running LightGBM (MOD-3)

Run from the project root. Same 13 features, encoding, and splits as MOD-2, so
only the algorithm changes. Results and the comparison with XGBoost:
`docs/findings.md` (MOD-3).

```bash
uv run python -m src.ensemble.lightgbm_model             # one run with LightGBM's default hyperparameters
uv run python -m src.ensemble.lightgbm_model --tune 90   # hyperopt search: 90 trials, then retrain the best
uv run python -m src.ensemble.lightgbm_model --params reports/lightgbm/mod3_metrics.json   # retrain the final model (7 leaves)
```

- **Default run:** trains once with `DEFAULT_HYPERPARAMS_DICT` (LightGBM's
  defaults). Takes about 10 seconds.
- **`--tune N`:** same as MOD-2, over LightGBM's equivalents: `learning_rate`,
  `num_leaves` (15–255, log scale), `min_child_samples` (20–2,000, log scale),
  `subsample`, `colsample_bytree`, `reg_lambda`, `max_cat_threshold`. Trials are
  logged as `MOD-3` runs tagged `hyperopt-trial`. 90 trials took about 1 hour 20
  minutes. The same long-run advice as MOD-2 applies (`caffeinate`, no branch
  switching, no other training at the same time).
- **Final model:** the search's best trial (26) used the smallest `num_leaves`
  allowed (15), so a boundary check retrained it with 4, 7, and 10 leaves; 7
  scored best. The final model is trial 26 with `num_leaves` 7, trained with
  `--params ... --set num_leaves=7`. `mod3_metrics.json` now stores 7, so
  `--params reports/lightgbm/mod3_metrics.json` alone reproduces it (best
  iteration 2,493, validation PR-AUC 0.331). Takes about 2–3 minutes.
- **Outputs:**
  - `reports/lightgbm/mod3_metrics.json`: run info, `train_seconds`, every hyperparameter used, `best_iteration` (stored 0-based like XGBoost's: trees kept = `best_iteration` + 1), `rounds_trained`, `train` and `validation` blocks
  - `reports/lightgbm/figures/mod3_pr_curve.png`, `mod3_thresholds.png`
  - `models/mod3_lightgbm.joblib`: the fitted model and its validation threshold
- Export the trials: `uv run python scripts/export_wandb_runs.py --run MOD-3 --tag hyperopt-trial --out reports/lightgbm/mod3_trials.csv`

## Running EXP-4 (historical late rate, `--late-rate`)

Run from the project root. Adds one feature, `late_rate` (the share of late
flights for the flight's `Origin` + `DepTimeBlk`, computed from train only;
out-of-fold on training rows, smoothed with α = 20), to the 13 base features.
Everything else stays as in MOD-1 / MOD-2. Code: `src/features/late_rate.py`.
Results: `docs/findings.md` (EXP-4).

```bash
uv run python -m src.baseline.logistic_regression --late-rate
uv run python -m src.ensemble.xgboost_model --params reports/xgboost/mod2_metrics.json --late-rate
```

- **XGBoost requires `--params`:** the experiment keeps MOD-2's tuned settings
  fixed, so `--late-rate` alone or with `--tune` is an error. Logistic
  regression has no tuned settings, so it needs no other flag.
- **Without `--late-rate`** both scripts behave exactly as before (checked:
  MOD-1 and MOD-2 reproduce their committed metrics).
- **Outputs** (never the `mod1_*` / `mod2_*` files):
  - `reports/logistic_regression/exp4_metrics.json`, `figures/exp4_pr_curve.png`, `figures/exp4_thresholds.png`, `models/exp4_logreg.joblib`
  - `reports/xgboost/exp4_metrics.json`, `figures/exp4_pr_curve.png`, `figures/exp4_thresholds.png`, `models/exp4_xgboost.joblib`
- **W&B:** runs `EXP-4-logistic-regression` and `EXP-4-xgboost`, tagged with
  the model and `13-plus-late-rate` (XGBoost also `final`). Each takes under a
  minute.

## Running EXP-2 (more training data, `--dataset`)

Run from the project root, after building the dataset's splits (see
[Build the processed data](#build-the-processed-data)). Trains XGBoost with
MOD-2's tuned settings on a longer training window; validation and test are the
same rows. Results: `docs/findings.md` (EXP-2). Verdict: not adopted, MOD-2 on
`data/processed` stays the tree model.

```bash
uv run python -m src.ensemble.xgboost_model --params reports/xgboost/mod2_metrics.json --dataset fullyear     # Jul 2024 - Mar 2026
uv run python -m src.ensemble.xgboost_model --params reports/xgboost/mod2_metrics.json --dataset aprjun2025   # Apr 2025 - Mar 2026 (run B)
```

- **`--dataset`** picks the processed folder (`DATA_FOLDERS` in
  `xgboost_model.py`, same names as `DATASETS` in `preprocess.py`). Default
  `base` = `data/processed`, so commands without it behave exactly as before
  (checked: `--params` reproduces MOD-2 exactly).
- **Requires `--params`:** the experiment keeps MOD-2's tuned settings fixed, so
  a non-base `--dataset` alone or with `--tune` is an error.
- **The flags pick the run.** Each row of `RUNS` in `xgboost_model.py` says
  which `--late-rate` / `--dataset` combination selects it, plus its tags and
  output files. A combination without a row (e.g. `--late-rate --dataset
  fullyear`) stops with an error before loading data; add a row to run it.
- **Outputs** (never the `mod2_*` / `exp4_*` files):
  - `fullyear`: `reports/xgboost/exp2_metrics.json`, `figures/exp2_pr_curve.png`, `figures/exp2_thresholds.png`, `models/exp2_xgboost.joblib`
  - `aprjun2025`: `reports/xgboost/exp2_aprjun_metrics.json`, `figures/exp2_aprjun_*.png`, `models/exp2_aprjun_xgboost.joblib`
  - Non-base runs add a `dataset` block (name and folder) to the metrics JSON
    and W&B config.
- **W&B:** runs `EXP-2` (tags `xgboost`, `13-baseline-features`,
  `21-training-months`, `final`) and `EXP-2-aprjun` (`12-training-months`
  instead). The `--set max_depth=6` / `7` runs on `fullyear` are tagged
  `depth-check` instead of `final`. The full-year run takes about 2–3 minutes.
- Adding `--set` to a `--dataset` run overwrites that dataset's `exp2*` files:
  commit them first, then `git restore reports/` afterwards (as for MOD-3's
  boundary check).

## Running a saved configuration (`--params`, `--set`)

Both tree models (`xgboost_model`, `lightgbm_model`) can train and save once
with hyperparameters read from a JSON file, instead of the defaults or a search.
Use it to reproduce a final model, or to run the tuned model with one setting
changed (experiments keep the tuned settings fixed).

```bash
# reproduce the final model from its metrics JSON
uv run python -m src.ensemble.lightgbm_model --params reports/lightgbm/mod3_metrics.json

# same, with one setting changed for this run only (repeat --set for more)
uv run python -m src.ensemble.lightgbm_model --params reports/lightgbm/mod3_metrics.json --set num_leaves=7
```

- **`--params PATH`:** a metrics JSON (its `hyperparameters` block is used) or a
  plain JSON object of hyperparameters. Fixed settings (`objective`,
  `n_estimators`, `random_state`, early stopping, ...) always come from the
  code, never from the file. Any tunable setting missing from the file falls back
  to the script's defaults.
- **`--set KEY=VALUE`:** overrides one tunable hyperparameter for this run only;
  the file is not changed. An unknown name (e.g. a typo like `num_leaf`) is an
  error rather than being silently ignored. Requires `--params`; cannot be
  combined with `--tune`.
- **Outputs:** the run is saved like a final model (metrics JSON, figures,
  model file) and logged to W&B tagged `final`. The file and overrides used are
  recorded as `params_source` in the metrics JSON and the W&B config.
- **Reproducing without changing anything:** each `--params` run rewrites the
  metrics JSON (same scores, new `date` and `train_seconds`). To check a result
  without logging a new W&B run or changing committed files, run with
  `WANDB_MODE=disabled` and then `git restore reports/`.

## Scoring the test split (EVAL-4)

Done once, on 2026-11-25; the results are final. `src/final_evaluation.py`
scores the frozen models (MOD-1 and MOD-2 final, MOD-3 reference only) on test
at the thresholds saved in their `.joblib` files. Results: `docs/findings.md`
(EVAL-4).

```bash
uv run python -m src.final_evaluation --split val    # dry run: must reproduce every saved validation metric; writes nothing
uv run python -m src.final_evaluation --split test   # the one real run (already done)
```

- `--split` is required, so the test run is always typed on purpose.
- Before scoring, it checks each model file against the SHA-256 in
  `FINAL_MODELS`; a retrained or overwritten model stops the script.
- Test encoding comes from the feature and category lists stored in each saved
  model, not from the model scripts, which this file never imports.
- `--split test` refuses to run if `reports/final/eval4_metrics.json` exists.
  Do not delete it to rerun: a second look at test after any change is tuning
  on test.
- **Outputs:** `reports/final/eval4_metrics.json` (validation and test blocks per
  model, unseen-category counts, hashes, library versions) and
  `reports/final/figures/eval4_val_vs_test.png`, `eval4_test_pr_curves.png`.
- **W&B:** three runs named `EVAL-4`, tagged with the model,
  `13-baseline-features`, `test`, and `final` or `reference`.

## Error analysis (EVAL-5)

Validation only; reads the frozen MOD-2 model (hash checked) and never the test
split. Not logged to W&B. Takes about 40 s (SHAP for every validation flight).
Results: `docs/findings.md` (EVAL-5).

```bash
uv run python -m src.error_analysis                      # groups below 1,000 validation flights are kept but not ranked
uv run python -m src.error_analysis --min-flights 2000   # a different cutoff
```

- **Outputs** in `reports/eval5/`: `eval5_feature_importance.csv` and
  `figures/eval5_feature_importance.png` (total gain on train next to mean
  |SHAP| on validation), `eval5_groups_Reporting_Airline.csv`,
  `eval5_groups_Origin.csv`, `eval5_groups_DepTimeBlk.csv` (per group: flights,
  late rate, mean probability, PR-AUC, `pr_auc_over_floor`), and
  `eval5_error_cases.csv` (confident and random false positives and false
  negatives at the frozen threshold, with their top SHAP contributions).
- Rerunning gives the same files: the random cases use a fixed seed.

## Tracking runs with Weights & Biases

Every model run is logged to the shared W&B project `flight-classifier` (team
`mattstaff0-texas-christian-university`), so every configuration tried is kept,
not just the last one. The JSON files in `reports/` stay the committed record of
final results.

**One-time setup (each teammate):**

1. Create a free account at [wandb.ai](https://wandb.ai) and ask to be invited to the team.
2. Log in from the project root and paste your API key when asked:
   ```bash
   uv run wandb login
   ```
   The key is saved in your user settings, not in the repo. Never commit it.

**Running without a login or network:** prefix any model command with
`WANDB_MODE=disabled`. All W&B calls do nothing; metrics, plots, and JSON are
still written.

```bash
WANDB_MODE=disabled uv run python -m src.baseline.logistic_regression
```

`WANDB_MODE=offline` instead saves the run in `wandb/` (ignored by git); upload
it later with `uv run wandb sync wandb/<run-folder>`.

**Adding W&B to a new model script:** import `log_to_wandb` and
`compute_loss` from `src.evaluation`. Every model reports a `train` block next
to `validation`. `log_to_wandb` sends whichever of `train`, `validation`, and
`test` are present to the summary (`SPLITS` in `src/evaluation.py`). Call
`log_to_wandb` once, after `save_json`. Do not import `wandb` in the model
script.

```python
training_probabilities = pipeline.predict_proba(training)[:, 1]
results["train"] = compute_loss(training_y, training_probabilities)
results["validation"] = validation_summary(metrics=metrics, floor=floor)
save_json(METRICS_PATH, results=results)
log_to_wandb(results=results,
             figures={"pr_curve": pr_curve_path, "thresholds": thresholds_path},
             tags=[MODEL_TAG, FEATURE_TAG])
```

- **Run name** comes from `results["run"]`: the issue ID (`MOD-1`, `MOD-2`, ...). An experiment run on more than one model
  adds the model so the runs can be told apart (`EXP-4-xgboost`,
  `EXP-4-logistic-regression`); W&B filters match the name exactly.
- **Config** is every key of `results` except the split blocks (`train`, `validation`, `test`). Put
  hyperparameters and run info in `results`.
- **Summary** is the `train` and `validation` blocks, nested, so W&B shows
  `train.log_loss`, `validation.log_loss`, `validation.pr_auc`, and so on.
  `train` holds only threshold-free scores (`log_loss`, `pr_auc`, `floor`,
  `pr_auc_over_floor`); the threshold, F1, and confusion matrix are chosen and
  reported on validation only.
- **Checking for overfitting:** compare `train.pr_auc_over_floor` with
  `validation.pr_auc_over_floor`, not raw log loss. Train (Jul–Mar) and
  validation (Apr) have different late rates, and log loss moves with the late
  rate: for MOD-1, validation log loss (0.479) is lower than train (0.504) only
  because April has fewer late flights.
- **Tags** are the model and the feature set, in kebab-case, spelled the same
  in every script (e.g. `logistic-regression`, `xgboost`, `lightgbm`,
  `13-baseline-features`, `13-plus-late-rate`), plus the run type: `default`,
  `hyperopt-trial`, `final`, `boundary-check`, or `depth-check`. The EVAL-4 test
  runs are tagged `test` plus `final` or `reference`. Runs on a
  training window other than Jul 2025 – Mar 2026 also get a data tag counting
  the training months (`21-training-months`, `12-training-months`). A `final` run that was
  replaced keeps its tags and gets `superseded` added (never delete it: its
  metrics JSON is in git history), so filter by `final` and exclude
  `superseded` to see the current final models.
- **Figure keys** are always `pr_curve` and `thresholds`, so plots from
  different models share a panel.
- **`history`** (optional) is a list of dicts, one per training round, e.g.
  `[{"train_logloss": 0.62, "val_logloss": 0.63}, ...]`; each key becomes a
  line chart. Use it for per-tree loss in boosted models.
- **Compare runs on `validation.*`, never on `val_logloss`.** With `history`,
  the summary holds two validation log losses that mean different things. W&B
  copies the *last* history step into the summary, so `val_logloss` is the
  final round trained, which with early stopping is always
  `early_stopping_rounds` past the best one (MOD-2 default run: 0.4788 at
  round 67). `validation.log_loss` is computed from the predictions of the
  model actually used, at the best iteration (0.4762 at round 17). Sort and
  filter the runs table by `validation.pr_auc`, `validation.f1`, and
  `validation.log_loss`.
- Training and experiment runs log only train and validation. The test split
  is scored and logged only by `src/final_evaluation.py` (EVAL-4, done once).
