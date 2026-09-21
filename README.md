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

The CSVs are not in git (~3 GB). Download them with:

```bash
python3 scripts/fetch_bts_data.py --out data/raw
```

This fetches all 12 months, skips any already present, and names them
`YYYY-MM-flight-data.csv`. Takes a few minutes.

## Scripts

- `scripts/fetch_bts_data.py` — download and extract the monthly data
- `scripts/utils.py` — row counts and late rates per month

`docs/readme.html` is the BTS field documentation for all 109 columns.
