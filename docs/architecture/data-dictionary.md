# Data Dictionary

**What this file describes**:

- Every column in the BTS Reporting Carrier On-Time Performance data (109 columns, definitions in `docs/readme.html`) and what we do with it: model feature, target, split key, filter-only, or dropped.
- The exact columns extracted into `data/processed` for training, validation, and testing.
- The reason each column is kept or dropped, so the feature set can be reviewed for data leakage. The rule: a model feature must be known **at booking time**.

**Status:** Confirmed by the team (PRE-1, 2026-10-11). Rows marked _pending_ depend on another open issue.

## Model Features

The processed files hold these columns as raw values. Encoding and scaling happen in the **model pipeline**, fit on the training split only — see "How each model encodes features" in [architecture-design.md](architecture-design.md). The two encoding columns are the plan for each model; a `—` means the column is not a model input.

| Column               | Description                                    | Type                    | Known at booking? | Logistic regression encoding | Gradient boosting encoding | Status / Notes                                                                                                                                                                                                     |
| -------------------- | ---------------------------------------------- | ----------------------- | ----------------- | ---------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DayofMonth`         | Day of month (1–31)                            | integer                 | yes               | one-hot                      | integer as-is              | Confirmed                                                                                                                                                                                                          |
| `DayOfWeek`          | Day of week (1 = Monday … 7 = Sunday)          | categorical (integer)   | yes               | one-hot                      | integer as-is              | Confirmed                                                                                                                                                                                                          |
| `Reporting_Airline`  | Unique carrier code                            | categorical             | yes               | one-hot                      | native categorical         | Confirmed                                                                                                                                                                                                          |
| `Origin`             | Origin airport code                            | categorical             | yes               | one-hot                      | native categorical         | Confirmed. Unseen airports in val/test: see PRE-6.                                                                                                                                                                 |
| `Dest`               | Destination airport code                       | categorical             | yes               | one-hot                      | native categorical         | Confirmed. Unseen airports in val/test: see PRE-6.                                                                                                                                                                 |
| `CRSDepTime`         | Scheduled departure time (local, hhmm)         | integer (hhmm)          | yes               | —                            | —                          | **Not a model input** (PRE-3): `DepTimeBlk` carries the same hour. Kept in the processed files for error analysis.                                                                                                 |
| `CRSArrTime`         | Scheduled arrival time (local, hhmm)           | integer (hhmm)          | yes               | —                            | —                          | **Not a model input** (PRE-3): `ArrTimeBlk` carries the same hour. Kept in the processed files for error analysis.                                                                                                 |
| `CRSElapsedTime`     | Scheduled elapsed time of flight (minutes)     | numeric                 | yes               | standardize                  | as-is                      | Confirmed. Use this for duration; red-eyes make `CRSArrTime − CRSDepTime` wrong.                                                                                                                                   |
| `Distance`           | Distance between airports (miles)              | numeric                 | yes               | standardize                  | as-is                      | Confirmed                                                                                                                                                                                                          |
| `Month`              | Month (1–12)                                   | categorical (integer)   | yes               | —                            | —                          | **Not a model input** on the current data (PRE-2): validation (Apr) and test (May–Jun) months never appear in train (Jul–Mar). Kept in `COLS` and the processed files; tested in EXP-3.                            |
| `DepTimeBlk`         | Scheduled departure hour block                 | categorical (19 values) | yes               | one-hot                      | ordinal 0–18               | Confirmed as the time-of-day feature (PRE-3). One overnight block `0001-0559` plus hourly blocks; matches the hour of `CRSDepTime` on every July 2025 row. Strings are zero-padded, so sorted order is time order. |
| `ArrTimeBlk`         | Scheduled arrival hour block                   | categorical (19 values) | yes               | one-hot                      | ordinal 0–18               | Confirmed as the time-of-day feature (PRE-3). Same as `DepTimeBlk`, for arrival.                                                                                                                                   |
| `OriginState`        | Origin airport state code                      | categorical             | yes               | one-hot                      | native categorical         | Confirmed. Coarser geography; helps when an airport is rare or unseen in train (PRE-6).                                                                                                                            |
| `DestState`          | Destination airport state code                 | categorical             | yes               | one-hot                      | native categorical         | Confirmed. Same as `OriginState`.                                                                                                                                                                                  |
| `OriginCityMarketID` | DOT city market ID (groups airports in a city) | categorical (integer)   | yes               | one-hot                      | native categorical         | Confirmed. 322 city markets vs 345 airports in July 2025. An ID, not a quantity — never use it as a number.                                                                                                        |
| `DestCityMarketID`   | DOT city market ID for the destination         | categorical (integer)   | yes               | one-hot                      | native categorical         | Confirmed. Same as `OriginCityMarketID`.                                                                                                                                                                           |

## Target

| Column     | Description                                           | Type | Notes                                    |
| ---------- | ----------------------------------------------------- | ---- | ---------------------------------------- |
| `ArrDel15` | Arrival delay indicator, 15 minutes or more (1 = Yes) | 0/1  | Null for cancelled and diverted flights. |

## Split Key

| Column       | Description              | Notes                                                                                                                                                                                |
| ------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `FlightDate` | Flight date (yyyy-mm-dd) | Assigns each row to train (2025-07 – 2026-03), validation (2026-04), or test (2026-05 – 2026-06). Kept in processed files for splitting and error analysis; **never a model input**. |

## Filter-Only Columns

These are read during preprocessing to apply the instance definition, then dropped. They must stay in the read list or the filter cannot run.

| Column      | Description                          | Filter rule                      | Why                                                                                                      |
| ----------- | ------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `Cancelled` | Cancelled flight indicator (1 = Yes) | keep rows where `Cancelled == 0` | Instance definition: we model completed, non-diverted flights. `ArrDel15` is null for these rows anyway. |
| `Diverted`  | Diverted flight indicator (1 = Yes)  | keep rows where `Diverted == 0`  | Same as above.                                                                                           |
| `ArrDel15`  | (target)                             | keep rows where not null         | A row without a label cannot be trained or scored.                                                       |

## Data Quality (PRE-4)

Measured on all 12 months **after** the instance filter, with `notebooks/PRE_4_Measure_NaN_counts.py`.

| Check | What was checked | Result | Action in preprocessing |
| ----- | ---------------- | ------ | ----------------------- |
| Missing values | Nulls in all 20 columns in `COLS` | 2 rows, both `CRSElapsedTime` (2026-03 train, 2026-06 test). Every other column is complete. | Drop (rule below) |
| Out-of-range | Day/weekday ranges, valid hhmm times, valid time blocks, `CRSElapsedTime > 0`, `Distance > 0`, `Origin != Dest`, `ArrDel15` in {0, 1} | 1 row: `CRSElapsedTime = −60` (2025-11, OO 3574 DTW → LEX, scheduled to arrive before it departs). All other rules: 0. | Drop (rule below) |
| Duplicates | Same flight twice, keyed on `FlightDate` + `Reporting_Airline` + `Flight_Number_Reporting_Airline` + `Origin` | 0 | None |

**Cleaning rule:** keep rows where `CRSElapsedTime > 0`. This one rule removes all 3 bad rows (a null fails the comparison too).

**Rows after cleaning:**

| Split | Months | Rows |
| ----- | ------ | ---- |
| Train | 2025-07 – 2026-03 | 5,105,534 |
| Validation | 2026-04 | 591,206 |
| Test | 2026-05 – 2026-06 | 1,199,540 |
| **Total** | | **6,896,280** (proposal: 6,896,283; −3 from the cleaning rule) |

**Takeaways for the report:**

- The data is almost complete: 3 bad rows out of 6.9M (0.00004%). No imputation is needed, so the model pipeline has no imputer.
- `isna()` alone would have missed the `−60` row; it only shows up in the range checks.
- Flight number alone does not identify a flight: date + carrier + flight number flags 100,654 "duplicates" in July 2025 alone, all of them multi-leg flights (e.g. AA 9 DFW → ABQ, then ABQ → DFW). Adding `Origin` to the key brings that to 0.

## Dropped Columns

| Group                      | Columns                                                                                                                                        | Reason                                                                                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Unseen time values         | `Year`, `Quarter`                                                                                                                              | Same problem as `Month`: train covers Q3, Q4, Q1; validation and test are entirely Q2.                                                      |
| Constant                   | `Flights`                                                                                                                                      | Always 1.0 (verified on all 631,428 July 2025 rows). Carries no information.                                                                |
| Redundant carrier IDs      | `DOT_ID_Reporting_Airline`, `IATA_CODE_Reporting_Airline`                                                                                      | Duplicates `Reporting_Airline`.                                                                                                             |
| Aircraft                   | `Tail_Number`                                                                                                                                  | Aircraft is assigned close to departure, not known at booking. 1,694 nulls in July 2025.                                                    |
| Flight number (raw)        | `Flight_Number_Reporting_Airline`                                                                                                              | Known at booking, but 22,001 carrier + flight-number combinations in July 2025 alone. Used only as a key for a derived feature (see below). |
| Redundant airport fields   | `OriginAirportID`, `OriginAirportSeqID`, `OriginCityName`, `OriginStateFips`, `OriginStateName`, `OriginWac`, and the matching `Dest*` columns | Duplicates `Origin` / `Dest` or the kept state / city-market columns.                                                                       |
| Redundant distance         | `DistanceGroup`                                                                                                                                | `Distance` in 250-mile bins.                                                                                                                |
| Post-departure (departure) | `DepTime`, `DepDelay`, `DepDelayMinutes`, `DepDel15`, `DepartureDelayGroups`, `TaxiOut`, `WheelsOff`                                           | **Leakage** — only known after the flight departs. Departure delay is strongly tied to arrival delay.                                       |
| Post-departure (arrival)   | `WheelsOn`, `TaxiIn`, `ArrTime`, `ArrDelay`, `ArrDelayMinutes`, `ArrivalDelayGroups`                                                           | **Leakage** — these are the outcome itself or computed from it.                                                                             |
| Actual flight times        | `ActualElapsedTime`, `AirTime`                                                                                                                 | **Leakage** — only known after the flight.                                                                                                  |
| Cancellation reason        | `CancellationCode`                                                                                                                             | Only exists for cancelled flights, which are filtered out.                                                                                  |
| Delay causes               | `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`                                                               | **Leakage** — only filled in for delayed flights, so they encode the label.                                                                 |
| Gate return                | `FirstDepTime`, `TotalAddGTime`, `LongestAddGTime`                                                                                             | **Leakage** — only known after the flight leaves the gate.                                                                                  |
| Diversion summary          | `DivAirportLandings`, `DivReachedDest`, `DivActualElapsedTime`, `DivArrDelay`, `DivDistance`                                                   | Only meaningful for diverted flights, which are filtered out.                                                                               |
| Diversion detail (×5)      | `Div1`–`Div5` × `Airport`, `AirportID`, `AirportSeqID`, `WheelsOn`, `TotalGTime`, `LongestGTime`, `WheelsOff`, `TailNum` (40 columns)          | Only meaningful for diverted flights, which are filtered out.                                                                               |

## Derived Features for the Ablation

These are not BTS columns. They are computed from the columns above and are candidates for the controlled comparison (model with vs. without them).

| Feature                    | Built from                                                                                                                             | Known at booking?                        | Leakage rule                                                                                                                                                                                                                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Historical late rate       | `ArrDel15` grouped by a key: `Origin` + departure hour (from the proposal), or `Reporting_Airline` + `Flight_Number_Reporting_Airline` | yes (it is a past statistic)             | Compute the rates from **train months only**, then look them up for validation and test rows. For the train rows themselves, use out-of-fold rates (each row's rate computed without that row's fold), otherwise a row's own label leaks into its feature. Unseen keys get the overall train late rate. |
| Days to nearest US holiday | `FlightDate` and the federal holiday calendar                                                                                          | yes (holiday dates are fixed in advance) | No label involved, so no fit step. Train covers Independence Day through Presidents' Day; test covers Memorial Day and Juneteenth.                                                                                                                                                                      |
