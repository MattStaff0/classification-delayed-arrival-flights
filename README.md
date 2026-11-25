# Flight Delay Classification

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
to `validation` (required: `log_to_wandb` raises `KeyError` without it). Call
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
- **Config** is every key of `results` except `train` and `validation`. Put
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
  `hyperopt-trial`, `final`, or `boundary-check`. A `final` run that was
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
- Only train and validation are logged. Never score or log the test split.
