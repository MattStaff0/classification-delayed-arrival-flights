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
uv run scripts/preprocess.py
```

Run from the project root. Filters to completed flights, drops the 3 bad rows,
splits by date, and writes `data/processed/{train,val,test}.parquet`. Stops with
an error if any check fails; re-running overwrites the files. See
`docs/architecture/data-dictionary.md` (Processed Files) for the contents.

## Scripts

- `scripts/fetch_bts_data.py` — download and extract the monthly data (AI-generated, see notice above)
- `scripts/utils.py` — row counts and late rates per month
- `scripts/preprocess.py` — build the train/val/test Parquet files

`docs/readme.html` is the BTS field documentation for all 109 columns.

## Running the baseline model

- Run from the project root
- `uv run python -m src.baseline.logistic_regression` - runs the logistic regression baseline, adds metrics to `reports/logistic_regression`

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

- **Run name** comes from `results["run"]`: the issue ID (`MOD-2`, `EXP-4`, ...).
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
  in every script (e.g. `logistic-regression`, `xgboost`, `13-baseline-features`,
  `late-rate`).
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
