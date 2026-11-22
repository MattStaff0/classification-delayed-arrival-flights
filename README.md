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
