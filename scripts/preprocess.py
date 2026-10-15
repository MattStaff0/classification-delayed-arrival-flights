import pandas as pd
from pathlib import Path
from utils import COLS

# dataset path contstants
PARENT_PATH = "data/raw"
DATASET_PATH = "*-flight-data.csv"
PROCESSED_PATH = "data/processed"

# training / validation / testing split blocks
VALIDATION_START = "2026-04-01"
TESTING_START = "2026-05-01"

CAT_COLS = ["Reporting_Airline", "Origin", "Dest", "DepTimeBlk", "ArrTimeBlk",
              "OriginState", "DestState"]

EXPECTED_COLS = ["FlightDate","Month","DayofMonth","DayOfWeek","Reporting_Airline",
        "Origin","Dest","CRSDepTime","CRSArrTime","CRSElapsedTime",
        "Distance","DepTimeBlk","ArrTimeBlk","OriginState","DestState",
        "OriginCityMarketID","DestCityMarketID","ArrDel15"]

EXPECTED_ROWS = {"train": 5_105_534, "val": 591_206, "test": 1_199_540}

UNSEEN_CHECK_COLS = ["Origin", "Dest", "Reporting_Airline", "OriginState", "DestState",
                     "OriginCityMarketID", "DestCityMarketID"]

def load_dataset_as_df(path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=COLS, engine="pyarrow")

def check_if_all_files_exist():
    files = [file for file in sorted(Path(PARENT_PATH).glob(DATASET_PATH))]

    # currently have 12 months of data will need to change this if we get more data
    assert len(files) == 12, f"Expected 12 monthly files, found {len(files)}"

    return True

def apply_instance_filter(df):
    return df.loc[(df["ArrDel15"].notna()) & (df["Cancelled"] == 0) & (df["Diverted"] == 0)]

def drop_bad_elapsed_time(df):
    # gets rid of the three bad rows
    return df.loc[df["CRSElapsedTime"] > 0]

def concatenate_data():
    files = [file for file in sorted(Path(PARENT_PATH).glob(DATASET_PATH))]
    dfs = []
    for file in files: 
        # get the dataframe
        temp = load_dataset_as_df(file)

        # get the correct instances
        temp = apply_instance_filter(temp)

        # get rid of rows the violate
        temp = drop_bad_elapsed_time(temp)

        # drop the cols we do not need
        temp = temp.drop(columns=["Diverted", "Cancelled"])

        # append it to the list of dfs
        dfs.append(temp)

    # concat all the dfs together into one huge df
    all_months = pd.concat(dfs, ignore_index=True)

    return all_months

def convert_types(df):
    # flgihtdate to pandas datetime not pythons for splits
    df["FlightDate"] = pd.to_datetime(df["FlightDate"]).astype("datetime64[ms]")  # ms: parquet has no seconds unit

    # convert arrdel15 to small int since its 1 or 0
    df["ArrDel15"] = df["ArrDel15"].astype("int8")

    # convert text columns to categories for Xgboost later
    df = df.astype({c: "category" for c in CAT_COLS})

    return df

def sort_by_flight_date(df):
    return df.sort_values("FlightDate", kind="stable")

def get_splits(df):
    # returns a tuple of our three splits

    # get the training split this is from 2025-07 -> 2026-03
    training_split = df.loc[df["FlightDate"] < VALIDATION_START]
    validation_split = df.loc[(df["FlightDate"] >= VALIDATION_START) & (df["FlightDate"] < TESTING_START)]
    testing_split = df.loc[df["FlightDate"] >= TESTING_START]

    training_split = training_split.reset_index(drop=True)
    validation_split = validation_split.reset_index(drop=True)
    testing_split = testing_split.reset_index(drop=True)


    return training_split, validation_split, testing_split

def write_parquet_splits(train_df, valid_df, test_df):
    # write out the three files as parquet
    # no try/except: if a write fails, the error should stop the script
    train_df.to_parquet(Path(PROCESSED_PATH) / "train.parquet", engine="pyarrow", index=False)
    valid_df.to_parquet(Path(PROCESSED_PATH) / "val.parquet", engine="pyarrow", index=False)
    test_df.to_parquet(Path(PROCESSED_PATH) / "test.parquet", engine="pyarrow", index=False)
    print(f"Successfully wrote files to {PROCESSED_PATH}! Great work :)")

# check splits against what we expect
def check_splits(train, val, test):
    # no nulls in any split
    assert train.notna().all().all(), f"There is a null in the training data, {train.isna().sum().to_string()}"
    assert val.notna().all().all(), f"There is a null in the validation data, {val.isna().sum().to_string()}"
    assert test.notna().all().all(), f"There is a null in the test data, {test.isna().sum().to_string()}"

    # make sure columns match what we need
    assert list(train.columns) == EXPECTED_COLS, f"Training set does not match expected columns: {list(set(train.columns) - set(EXPECTED_COLS))}"
    assert list(val.columns) == EXPECTED_COLS, f"Validation set does not match expected columns: {list(set(val.columns) - set(EXPECTED_COLS))}"
    assert list(test.columns) == EXPECTED_COLS, f"Test set does not match expected columns: {list(set(test.columns) - set(EXPECTED_COLS))}"

    # ArrDel15 only contains 0 or 1
    assert train["ArrDel15"].isin([0, 1]).all(), f"Arrdel15 is not all 0 or 1 - training"
    assert val["ArrDel15"].isin([0, 1]).all(), f"Arrdel15 is not all 0 or 1 - validation"
    assert test["ArrDel15"].isin([0, 1]).all(), f"Arrdel15 is not all 0 or 1 - test"

    # make sure our counts match what we expect
    assert len(train) == EXPECTED_ROWS.get("train"), f"Training data does not match expected size"
    assert len(val) == EXPECTED_ROWS.get("val"), f"Validation data does not match expected size"
    assert len(test) == EXPECTED_ROWS.get("test"), f"Test data does not match expected size"

    # make sure splits are accurate to month
    assert train["FlightDate"].max() < val["FlightDate"].min()
    assert val["FlightDate"].max() < test["FlightDate"].min()

def print_report(train, val, test):
    splits = {"train": train, "val": val, "test": test}

    # PRE-5: date range, rows, and late rate per split (ArrDel15 is 0/1 so mean is the late rate)
    print("Split report")
    print(f"{'split':<6} {'first date':<11} {'last date':<11} {'rows':>10} {'late rate':>10}")
    for name, split in splits.items():
        print(f"{name:<6} {split['FlightDate'].min().date()!s:<11} {split['FlightDate'].max().date()!s:<11} "
              f"{len(split):>10,} {split['ArrDel15'].mean():>10.3f}")
    total = sum(len(split) for split in splits.values())
    print(f"{'total':<6} {'':<11} {'':<11} {total:>10,}")

    # PRE-6: rows in val/test whose category value never appears in train
    print("\nRows with a value never seen in train")
    print(f"{'column':<20} {'val':>8} {'test':>8}")
    for col in UNSEEN_CHECK_COLS:
        seen = set(train[col].unique())
        unseen_val = int((~val[col].isin(seen)).sum())
        unseen_test = int((~test[col].isin(seen)).sum())
        print(f"{col:<20} {unseen_val:>8,} {unseen_test:>8,}")
    print()

def check_written_files(train, val, test):
    # read each file back and confirm it matches what was written
    for name, split in [("train", train), ("val", val), ("test", test)]:
        written = pd.read_parquet(Path(PROCESSED_PATH) / f"{name}.parquet")
        assert len(written) == len(split), f"{name}.parquet has {len(written):,} rows, expected {len(split):,}"
        assert written.dtypes.equals(split.dtypes), f"{name}.parquet dtypes changed:\n{written.dtypes.to_string()}"
    print("Read-back check passed: row counts and dtypes match.")

def main():
    # check if all the files exist before we run anything
    check_if_all_files_exist()

    df = concatenate_data()
    df = convert_types(df)
    df = sort_by_flight_date(df)
    train, val, test = get_splits(df)

    check_splits(train, val, test)
    print_report(train, val, test)
    write_parquet_splits(train, val, test)
    check_written_files(train, val, test)


if __name__ == "__main__":
    main()