import pandas as pd
from pathlib import Path


# Copied from COLS in scripts/utils.py; update both if the column list changes.
COLS = ["FlightDate","Month","DayofMonth","DayOfWeek","Reporting_Airline",
        "Origin","Dest","CRSDepTime","CRSArrTime","CRSElapsedTime",
        "Distance","DepTimeBlk","ArrTimeBlk","OriginState","DestState",
        "OriginCityMarketID","DestCityMarketID","ArrDel15","Cancelled","Diverted", "Flight_Number_Reporting_Airline"]
# Flight_Number_Reporting_Airline used to check for duplicate flights -> not in our column list see data-dict for actual column list


VALID_TIME_BLOCKS = ["0001-0559"] + [f"{h:02d}00-{h:02d}59" for h in range(6, 24)]
DUP_CHECK_FLIGHT_KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin"]

def get_data_set(path):
    return pd.read_csv(path, usecols=COLS, engine="pyarrow")

def get_filtered_df(path):
    # get the dataset
    df = get_data_set(path)

    # filter on the rows that we are going to use
    df = df.loc[(df["ArrDel15"].notna()) & (df["Cancelled"] == 0) & (df["Diverted"] == 0)]

    return df

def count_nulls(df):
    # null count per column for one month, plus the row count as the denominator
    counts = df.isna().sum()
    counts["rows"] = len(df)
    return counts

def summarize_nulls(missing_table):
    # per-column summary over all months; "rows" is the denominator, not a column
    totals = missing_table.sum(axis=1)
    return pd.DataFrame({
        "total_nulls": totals.drop("rows"),
        "pct_missing": totals.drop("rows") / totals["rows"] * 100,
    })

def valid_hhmm(s):
    return s.between(1, 2400) & (s % 100 < 60)

def count_out_of_range(df):
    # each rule is True for rows that are OK
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

def count_duplicates(df):
    return int(df.duplicated(subset=DUP_CHECK_FLIGHT_KEY).sum())

def run_checks():
    # read each month once and run every check on it
    nulls, out_of_range, duplicates = {}, {}, None

    for file in sorted(Path("data/raw").glob("*-flight-data.csv")):
        df = get_filtered_df(file)
        month = file.stem[:7]  # "2025-07"

        nulls[month] = count_nulls(df)
        out_of_range[month] = count_out_of_range(df)
        duplicates = count_duplicates(df)

    return pd.DataFrame(nulls), pd.DataFrame(out_of_range), duplicates

def main():
    missing_table, out_of_range_table, duplicates = run_checks()

    print("Nulls per month (after filtering):")
    print(missing_table.to_string())
    print("\nMissing values over all months:")
    print(summarize_nulls(missing_table).to_string(float_format="{:.6f}".format))

    print("\nOut-of-range rows per month:")
    print(out_of_range_table.to_string())
    print("\nOut-of-range rows over all months:")
    print(out_of_range_table.sum(axis=1).to_string())

    print("\nDuplicate rows:")
    print(duplicates)

if __name__ == "__main__":
    main()
