import argparse
import csv
import pandas as pd
from pathlib import Path

# EXP-2: checks the new raw months (Jul 2024 - Jun 2025) before they are used for training.
# Read-only: writes nothing. Run from the project root:
#   uv run python notebooks/EXP_2_Check_new_months.py


# Copied from COLS in scripts/utils.py; update both if the column list changes.
COLS = ["FlightDate","Month","DayofMonth","DayOfWeek","Reporting_Airline",
        "Origin","Dest","CRSDepTime","CRSArrTime","CRSElapsedTime",
        "Distance","DepTimeBlk","ArrTimeBlk","OriginState","DestState",
        "OriginCityMarketID","DestCityMarketID","ArrDel15","Cancelled","Diverted"]
# Flight_Number_Reporting_Airline only for the duplicate check (same as PRE-4), dropped after
DUP_CHECK_FLIGHT_KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin"]

NEW_RAW_PATH = "data/raw_2024"
OLD_RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
DATASET_PATH = "*-flight-data.csv"

VALID_TIME_BLOCKS = ["0001-0559"] + [f"{h:02d}00-{h:02d}59" for h in range(6, 24)]

# columns where a value never seen in the current train would become a new category in encode()
CATEGORY_CHECK_COLS = ["Reporting_Airline", "Origin", "Dest", "OriginState", "DestState",
                       "OriginCityMarketID", "DestCityMarketID", "DepTimeBlk", "ArrTimeBlk"]

def get_header(path):
    # first line of the csv only, so we can compare the full 110 column layout
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        return next(csv.reader(handle))

def get_data_set(path):
    return pd.read_csv(path, usecols=COLS + ["Flight_Number_Reporting_Airline"], engine="pyarrow")

def apply_instance_filter(df):
    # same filter as scripts/preprocess.py
    return df.loc[(df["ArrDel15"].notna()) & (df["Cancelled"] == 0) & (df["Diverted"] == 0)]

def drop_bad_elapsed_time(df):
    # same cleaning rule as scripts/preprocess.py
    return df.loc[df["CRSElapsedTime"] > 0]

def valid_hhmm(s):
    return s.between(1, 2400) & (s % 100 < 60)

def count_out_of_range(df):
    # same rules as PRE-4, each rule is True for rows that are OK
    rules = {
        "DayOfWeek 1-7": df["DayOfWeek"].between(1, 7),
        "DayofMonth 1-31": df["DayofMonth"].between(1, 31),
        "CRSDepTime valid": valid_hhmm(df["CRSDepTime"]),
        "CRSArrTime valid": valid_hhmm(df["CRSArrTime"]),
        "DepTimeBlk valid": df["DepTimeBlk"].isin(VALID_TIME_BLOCKS),
        "ArrTimeBlk valid": df["ArrTimeBlk"].isin(VALID_TIME_BLOCKS),
        "CRSElapsedTime > 0": df["CRSElapsedTime"] > 0,
        "Distance > 0": df["Distance"] > 0,
        "Origin != Dest": df["Origin"] != df["Dest"],
        "ArrDel15 is 0/1": df["ArrDel15"].isin([0, 1]),
    }
    # ~ flips to rows that broke rule, sum counts them
    return pd.Series({name: int((~ok).sum()) for name, ok in rules.items()})

def check_month(file, reference_header, reference_dtypes):
    # one row of the per-month table, plus the kept rows for the later checks
    month = file.stem[:7]  # "2024-07"
    raw = get_data_set(file)
    filtered = apply_instance_filter(raw)
    kept = drop_bad_elapsed_time(filtered)

    summary = {
        "header same": get_header(file) == reference_header,
        "dtypes same": bool((raw[COLS].dtypes == reference_dtypes).all()),
        "raw rows": len(raw),
        "cancelled": int((raw["Cancelled"] == 1).sum()),
        "diverted": int((raw["Diverted"] == 1).sum()),
        "nulls after filter": int(filtered[COLS].isna().sum().sum()),
        "bad elapsed": len(filtered) - len(kept),
        "wrong month": int((~kept["FlightDate"].astype(str).str.startswith(month)).sum()),
        "duplicates": int(kept.duplicated(subset=DUP_CHECK_FLIGHT_KEY).sum()),
        "kept rows": len(kept),
        "late rate": round(kept["ArrDel15"].mean(), 3),
    }

    # drop the filter-only columns like preprocess.py does
    kept = kept.drop(columns=["Cancelled", "Diverted", "Flight_Number_Reporting_Airline"])
    return month, summary, count_out_of_range(filtered), kept

def run_checks(new_raw_path, old_raw_path):
    # the first current month is the reference for the header and the dtypes
    reference_file = sorted(Path(old_raw_path).glob(DATASET_PATH))[0]
    reference_header = get_header(reference_file)
    reference_dtypes = pd.read_csv(reference_file, usecols=COLS, engine="pyarrow").dtypes

    files = sorted(Path(new_raw_path).glob(DATASET_PATH))
    assert len(files) == 12, f"Expected 12 new monthly files, found {len(files)}"

    # read each month once and run every check on it
    summaries, out_of_range, kept_months = {}, {}, []
    for file in files:
        month, summary, month_out_of_range, kept = check_month(file, reference_header, reference_dtypes)
        summaries[month] = summary
        out_of_range[month] = month_out_of_range
        kept_months.append(kept)

    new_months = pd.concat(kept_months, ignore_index=True)
    return pd.DataFrame(summaries).T, pd.DataFrame(out_of_range), new_months

def unseen_categories(new_months, train):
    # values in the new months that the current train never has
    rows = {}
    for col in CATEGORY_CHECK_COLS:
        seen = set(train[col].astype(str).unique())
        values = new_months[col].astype(str)
        unseen = sorted(set(values.unique()) - seen)
        rows[col] = {"values": len(unseen), "rows": int(values.isin(unseen).sum()), "which": ", ".join(unseen)}
    return pd.DataFrame(rows).T

def unseen_test_in_new_months(new_months, train, test):
    # PRE-6: test values unseen in the current train -> how many rows of them the new months add
    rows = {}
    for col in CATEGORY_CHECK_COLS:
        seen = set(train[col].astype(str).unique())
        for value in sorted(set(test[col].astype(str).unique()) - seen):
            rows[f"{col} {value}"] = int((new_months[col].astype(str) == value).sum())
    return pd.Series(rows, dtype="int64")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-raw", default=NEW_RAW_PATH, help="folder with the new months")
    parser.add_argument("--old-raw", default=OLD_RAW_PATH, help="folder with the current months (reference header)")
    parser.add_argument("--processed", default=PROCESSED_PATH, help="folder with the current train/val/test splits")
    args = parser.parse_args()

    month_table, out_of_range_table, new_months = run_checks(args.new_raw, args.old_raw)
    train = pd.read_parquet(Path(args.processed) / "train.parquet")
    val = pd.read_parquet(Path(args.processed) / "val.parquet")
    test = pd.read_parquet(Path(args.processed) / "test.parquet")

    print("Per month (filter = ArrDel15 not null, Cancelled 0, Diverted 0; clean = CRSElapsedTime > 0):")
    print(month_table.to_string())

    print("\nOut-of-range rows per month (after the filter, before cleaning):")
    print(out_of_range_table.to_string())

    # what the full-year train will look like (new months + current train)
    full_rows = len(new_months) + len(train)
    full_late = (new_months["ArrDel15"].sum() + train["ArrDel15"].sum()) / full_rows
    spring = new_months["FlightDate"].astype(str).str[:7].between("2025-04", "2025-06")
    print("\nTrain sizes:")
    print(f"new months            {len(new_months):>12,} rows, late rate {new_months['ArrDel15'].mean():.4f}")
    print(f"current train         {len(train):>12,} rows, late rate {train['ArrDel15'].mean():.4f}")
    print(f"full-year train       {full_rows:>12,} rows, late rate {full_late:.4f}")
    print(f"Apr-Jun 2025 only     {int(spring.sum()):>12,} rows, late rate {new_months.loc[spring, 'ArrDel15'].mean():.4f}")

    print("\nValues in the new months never seen in the current train:")
    print(unseen_categories(new_months, train).to_string())

    # airlines per month, so a carrier that starts or stops reporting shows up
    print("\nRows per airline per month (new months):")
    per_airline = pd.crosstab(new_months["Reporting_Airline"].astype(str), new_months["FlightDate"].astype(str).str[:7])
    print(per_airline.to_string())
    print("\nRows per airline in the current validation / test:")
    print(pd.DataFrame({"val": val["Reporting_Airline"].astype(str).value_counts(),
                        "test": test["Reporting_Airline"].astype(str).value_counts()}).fillna(0).astype(int).to_string())

    print("\nTest values unseen in the current train -> rows in the new months:")
    print(unseen_test_in_new_months(new_months, train, test).to_string())

if __name__ == "__main__":
    main()
