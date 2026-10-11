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
- **Schedule:** `CRSDepTime`, `CRSArrTime`, `DepTimeBlk`, `ArrTimeBlk`, `CRSElapsedTime`, `Distance`
- **Calendar:** `DayOfWeek`, `DayofMonth`, `Month` (pending PRE-2)
- **Target:** `ArrDel15`
- **Derived, for the ablation:** historical late rate (computed from train months only) and days to nearest US holiday

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
