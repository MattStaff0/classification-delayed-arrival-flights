# What this file is

- This file lays out a proposed architecture for creating a model that predicts if a flight will have a delayed arrival.
- This file includes a diagram of what the flow of our experiments for testing out different models.

# What Models are we testing?

## Baseline Model

- The baseline model that we are going to use is a simple KNN algorithm or a logistic regression

### Considerations

- A logisitc regression model uses an error based approach in which we use binary cross-entropy loss and error to train the model to make better predicitions.
  - Hyperparameters: # training iterations, learning rate, regularization constant, convergance limit

- A KNN uses a similarity approach in which we choose the _k_ nearest instances and use either majority voting or weighted voting to choose which class a query belongs to.
  - Hyperparameters: K, distance metric used, wieght function for weighted KNN, search algorithm (K-dim tree)

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

- **Where it happens:** in the model pipeline (scikit-learn `Pipeline` + `ColumnTransformer`), not in `preprocess.py`. The processed files hold raw values.
- **Fit on train only:** every encoder and scaler is fit on the training split, then applied unchanged to validation and test. Fitting on all the data would let validation/test values shape the encoding (leakage).
- **Logistic regression** learns one weight per input column:
  - Categorical columns → one-hot (`OneHotEncoder(handle_unknown="ignore")`), so each category gets its own weight. Integer codes would force a straight-line effect — e.g. late rate rises through the day and falls after 20:00 (July 2025), which one weight on a 0–18 code cannot fit.
  - Numeric columns → standardized (`StandardScaler`), so the solver converges and regularization treats features on a comparable scale.
- **Gradient boosting** splits on thresholds:
  - No scaling needed.
  - Ordered categories (`DepTimeBlk`, `ArrTimeBlk`) → ordinal codes 0–18, so one split can express "departures after 15:00".
  - Unordered categories (airline, airports, states, city markets) → the library's native categorical support.
- **Unseen categories** in validation/test (PRE-6): logistic regression ignores them (all-zero one-hot row); for gradient boosting, confirm how the chosen library (XGBoost or LightGBM) handles an unseen category before relying on it.

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
