# Data Dictionary

**What this file describes**:

- Every column in the BTS Reporting Carrier On-Time Performance data (109 columns, definitions in `docs/readme.html`) and what we do with it: model feature, target, split key, filter-only, or dropped.
- The exact columns extracted into `data/processed` for training, validation, and testing.
- The reason each column is kept or dropped, so the feature set can be reviewed for data leakage. The rule: a model feature must be known **at booking time**.

**Status:** Confirmed by the team (PRE-1, 2026-10-11). Rows marked _pending_ depend on another open issue.

## Model Features

| Column               | Description                                    | Type                  | Known at booking? | Status / Notes                                                                                                                     |
| -------------------- | ---------------------------------------------- | --------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `DayofMonth`         | Day of month (1–31)                            | integer               | yes               | Confirmed                                                                                                                          |
| `DayOfWeek`          | Day of week (1 = Monday … 7 = Sunday)          | categorical (integer) | yes               | Confirmed                                                                                                                          |
| `Reporting_Airline`  | Unique carrier code                            | categorical           | yes               | Confirmed                                                                                                                          |
| `Origin`             | Origin airport code                            | categorical           | yes               | Confirmed                                                                                                                          |
| `Dest`               | Destination airport code                       | categorical           | yes               | Confirmed                                                                                                                          |
| `CRSDepTime`         | Scheduled departure time (local, hhmm)         | integer (hhmm)        | yes               | _Pending PRE-3_ — converted to departure hour; raw hhmm is not used as a number                                                    |
| `CRSArrTime`         | Scheduled arrival time (local, hhmm)           | integer (hhmm)        | yes               | _Pending PRE-3_ — converted to arrival hour                                                                                        |
| `CRSElapsedTime`     | Scheduled elapsed time of flight (minutes)     | numeric               | yes               | Confirmed                                                                                                                          |
| `Distance`           | Distance between airports (miles)              | numeric               | yes               | Confirmed                                                                                                                          |
| `Month`              | Month (1–12)                                   | categorical (integer) | yes               | **Not a model input** on the current data (PRE-2): validation (Apr) and test (May–Jun) months never appear in train (Jul–Mar). Kept in `COLS` and the processed files; tested in EXP-3                                       |
| `DepTimeBlk`         | Scheduled departure hour block                 | categorical           | yes               | 19 values in July 2025: one overnight block `0001-0559` plus hourly blocks. Overlaps the departure hour from `CRSDepTime` (PRE-3). |
| `ArrTimeBlk`         | Scheduled arrival hour block                   | categorical           | yes               | Same as `DepTimeBlk`, for arrival. Overlaps the arrival hour from `CRSArrTime` (PRE-3).                                            |
| `OriginState`        | Origin airport state code                      | categorical           | yes               | Coarser geography; helps when an airport is rare or unseen in train (PRE-6).                                                       |
| `DestState`          | Destination airport state code                 | categorical           | yes               | Same as `OriginState`.                                                                                                             |
| `OriginCityMarketID` | DOT city market ID (groups airports in a city) | categorical (integer) | yes               | 322 city markets vs 345 airports in July 2025. An ID, not a quantity — must be encoded as a category.                              |
| `DestCityMarketID`   | DOT city market ID for the destination         | categorical (integer) | yes               | Same as `OriginCityMarketID`.                                                                                                      |

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
