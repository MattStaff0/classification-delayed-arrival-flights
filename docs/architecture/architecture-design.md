# What this file is

- This file lays out a proposed architecture for creating a model that predicts if a flight will have a delayed arrival.
- This file includes a diagram of what the flow of our experiments for testing out different models.

# What Models are we testing?

## Baseline Model

- The baseline model that we are going to use is a logistic regression algorithm

- A logisitc regression model uses an error based approach in which we use binary cross-entropy loss and error to train the model to make better predicitions.
  - Hyperparameters: # training iterations, learning rate, regularization constant, convergance limit

## Models to experiment - Ensemble Decision Tree Based Algorithms

### Bagging Family - Parallel Process

1. Random Forest model
   - Takes boostrap samples from dataset and creates many trees in parallel

### Boosting Family - Iterative Process

2. Adaboost (Adaptive Boosting)
   - Corrects mistakes by heavily weighting data points that the **prior** tree got wrong, so the next tree pays more attention to them
3. XGBoost (Extreme Gradient Boosting)
   - Optimized form of gradient boosting, instead of just adjusting the weights, use **gradient descent** and **regularization** (to prevent overfitting).
   - Builds trees _level by level_
4. LightGBM (Light Gradient Boosting Model)
   - Grows trees _leaf-wise_ so it is lighter on compute

## What features are we using?

Only columns known at booking time are model inputs. The full column-by-column decision (kept, filter-only, dropped, and why) is in [data-dictionary.md](data-dictionary.md).

- **Carrier and route:** `Reporting_Airline`, `Origin`, `Dest`, `OriginState`, `DestState`, `OriginCityMarketID`, `DestCityMarketID`
- **Schedule:** `DepTimeBlk`, `ArrTimeBlk` (time of day, PRE-3), `CRSElapsedTime`, `Distance`. `CRSDepTime`/`CRSArrTime` are read but are not model inputs.
- **Calendar:** `DayOfWeek`, `DayofMonth`, `Month` (read but not a model input on the current data, PRE-2; tested in EXP-3)
- **Target:** `ArrDel15`
- **Derived, for the ablation:** historical late rate (computed from train months only), days to nearest US holiday, historical weather delay rate (computed from the training months)

## How each model encodes features

Both models get the **same features**; only the encoding differs, because the models use their inputs differently. The per-column plan is in the encoding columns of [data-dictionary.md](data-dictionary.md).

- **Where it happens:** in each model script, not in `preprocess.py`. The processed files hold raw values. Logistic regression uses a scikit-learn `Pipeline` + `ColumnTransformer`; XGBoost uses `encode()` in `src/ensemble/xgboost_model.py` (pandas `category` columns).
- **Fit on train only:** every encoder and scaler is fit on the training split, then applied unchanged to validation and test. Fitting on all the data would let validation/test values shape the encoding (leakage). For pandas categories this means validation and test are re-coded to **train's** category list (`pd.Categorical(..., categories=train[col].cat.categories)`); otherwise each file numbers its categories differently and the same code means a different airport.
- **Logistic regression** learns one weight per input column:
  - Categorical columns → one-hot (`OneHotEncoder(handle_unknown="ignore")`), so each category gets its own weight. Integer codes would force a straight-line effect — e.g. late rate rises through the day and falls after 20:00 (July 2025), which one weight on a 0–18 code cannot fit.
  - Numeric columns → standardized (`StandardScaler`), so the solver converges and regularization treats features on a comparable scale.
- **Gradient boosting** splits numeric columns on thresholds ("Distance ≤ 800") and categorical columns into groups ("Origin in {ATL, ORD, ...}"):
  - No scaling needed: a threshold split puts the same flights on each side whatever the units.
  - Numeric columns (`CRSElapsedTime`, `Distance`, `DayofMonth`, `DayOfWeek`) → as-is.
  - All categorical columns, **including the time blocks** (`DepTimeBlk`, `ArrTimeBlk`) → pandas `category` with the library's native categorical support (`enable_categorical=True` in XGBoost). City market IDs are stored as integers but are labels, not quantities, so they are converted to `category` too.
  - **Change from the original plan (MOD-2):** the time blocks were planned as ordinal codes 0–18, so one split could express "departures after 15:00". MOD-2 kept them as categories for a simpler encoding; a category split can still group adjacent blocks, but has to learn their order from the data. EXP-9 tests whether ordinal codes do better.
  - Native categorical splits avoid one-hot encoding, which for ~1,500 category columns × 5.1M rows would need ~60 GB in memory and could only split off one category at a time.
- **Unseen categories** in validation/test (PRE-6): logistic regression ignores them (all-zero one-hot row). For XGBoost, re-coding to train's category list turns an unseen value into a missing value, which follows each split's learned default branch, so the row is predicted from its other features (checked in MOD-2). The re-coding is required: XGBoost 3.4 rejects a category label it did not see in training. Validation has none; test has 51/48 rows (EVAL-4).

## How are we using our training / validation / testing splits

We are going to use Out-of-Time sampling for our training // validation // testing splits

- Training Split: July, 2025 - March, 2026
- Validation Split: April 2026
- Testing Split: May, 2026 - June, 2026

## How are we going to evaluate our model?

### Metrics

1. Precision
2. Recall
3. F1-Score
4. Precision-Recall Area Under the Curve
