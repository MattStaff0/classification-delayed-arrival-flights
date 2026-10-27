# Flight Delay Classification

Predicting whether a scheduled flight will arrive 15+ minutes late (`ArrDel15`),
using only information known at booking time. Data is BTS Reporting Carrier
On-Time Performance, July 2025 – June 2026.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

Download the selected data from the BTS Website.

## Build the processed data

```bash
python3 scripts/preprocess.py
```

Run from the project root. Filters to completed flights, drops the 3 bad rows,
splits by date, and writes `data/processed/{train,val,test}.parquet`. Stops with
an error if any check fails; re-running overwrites the files. See
`docs/architecture/data-dictionary.md` (Processed Files) for the contents.

## Scripts

- `scripts/fetch_bts_data.py` — download and extract the monthly data
- `scripts/utils.py` — row counts and late rates per month
- `scripts/preprocess.py` — build the train/val/test Parquet files

`docs/readme.html` is the BTS field documentation for all 109 columns.

## Running the baseline model

- Run from the project root
- `python -m src.baseline.logistic_regression` - runs the logistic regression baseline, adds metrics to `reports/logistic_regression`
