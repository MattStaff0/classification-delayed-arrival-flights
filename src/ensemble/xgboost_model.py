from datetime import date

import pandas as pd
from pathlib import Path
import xgboost as xgb
from src.evaluation import evaluate, print_report, plot_pr_curve, plot_thresholds, save_model, validation_summary, save_json, log_to_wandb, compute_loss

# Constants

ORDINAL_COLUMNS = ["DepTimeBlk", "ArrTimeBlk"]
CATEGORICAL_COLUMNS = ["Origin", "Dest", "OriginCityMarketID", "DestCityMarketID", "OriginState", "DestState", "Reporting_Airline"]
NUMERIC_COLUMNS = ["CRSElapsedTime", "Distance", "DayofMonth", "DayOfWeek"]
FEATURES = ORDINAL_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET = "ArrDel15"

MODEL_PATH = "models/mod2_xgboost.joblib"
PREPROCESS_PATH = "data/processed"
FIGURES_PATH = "reports/xgboost/figures"
METRICS_PATH = "reports/xgboost/mod2_metrics.json"
WANDB_MODEL_TAG = "xgboost"
WANDB_FEATURE_TAG = "13-baseline-features"

# training setup
RANDOM_STATE = 42
TREE_METHOD = "hist"
EVAL_METRIC = "logloss"
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS = 2000     # upper limit; early stopping picks the actual tree count

# hyperparameters (XGBoost 3.4 defaults for now; tuned later with hyperopt)
LEARNING_RATE = 0.3
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 1
SUBSAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0
REG_LAMBDA = 1.0
REG_ALPHA = 0.0
GAMMA = 0.0
MAX_CAT_TO_ONEHOT = 4
MAX_CAT_THRESHOLD = 64


def load_splits():
    training_split = pd.read_parquet(Path(PREPROCESS_PATH) / "train.parquet")
    validation_split = pd.read_parquet(Path(PREPROCESS_PATH) / "val.parquet")
    return training_split, validation_split

def split_target_from_data(training, validation):
    training_y = training.pop(TARGET)
    validation_y = validation.pop(TARGET)
    return training, training_y, validation, validation_y

def encode(train, val):
    # grab only the 13 features
    train = train[FEATURES]
    val = val[FEATURES]

    # make the marketID's into categories
    train["OriginCityMarketID"] = train["OriginCityMarketID"].astype("category")
    train["DestCityMarketID"] = train["DestCityMarketID"].astype("category")

    for col in CATEGORICAL_COLUMNS + ORDINAL_COLUMNS:
        val[col] = pd.Categorical(val[col], categories=train[col].cat.categories)


    return train, val

def build_model():
    model = xgb.XGBClassifier(tree_method=TREE_METHOD,
                              enable_categorical=True,
                              eval_metric=EVAL_METRIC,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                              n_estimators=N_ESTIMATORS,
                              random_state=RANDOM_STATE,
                              learning_rate=LEARNING_RATE,
                              max_depth=MAX_DEPTH,
                              min_child_weight=MIN_CHILD_WEIGHT,
                              subsample=SUBSAMPLE,
                              colsample_bytree=COLSAMPLE_BYTREE,
                              reg_lambda=REG_LAMBDA,
                              reg_alpha=REG_ALPHA,
                              gamma=GAMMA,
                              max_cat_to_onehot=MAX_CAT_TO_ONEHOT,
                              max_cat_threshold=MAX_CAT_THRESHOLD)
    return model

def main():
    # load the splits
    training, validation = load_splits()

    # split the target off the datasets
    training, training_y, validation, validation_y = split_target_from_data(training=training, validation=validation)

    # encode the columns to categories for xgboost
    training, validation = encode(train=training, val=validation)

    # make the model
    model = build_model()

    # fit the model, verbose will report every 100 rounds the rounds are based on N_ESTIMATORS
    model.fit(training, training_y, eval_set=[(training, training_y), (validation, validation_y)], verbose=100)

    # print where the model stopped and what the log loss at that round
    best_iteration = model.best_iteration
    best_log_loss = model.best_score
    print(f'Best Iteration: {best_iteration}, Best Log Loss: {best_log_loss}')

    # training and validation probabilites
    training_probabilities = model.predict_proba(training)[:, 1]
    validation_probabilities = model.predict_proba(validation)[:, 1]
    validation_floor = validation_y.mean()

    validation_metrics = evaluate(validation_y, validation_probabilities)
    training_metrics = compute_loss(training_y, training_probabilities)
    print_report(validation_metrics, validation_floor)
    print(f'Training PR-auc-over-floor: {training_metrics["pr_auc_over_floor"]}')

    # create the figures 
    figure_directory = Path(FIGURES_PATH)
    figure_directory.mkdir(parents=True, exist_ok=True)

    pr_curve_path = figure_directory / "mod2_pr_curve.png"
    thresholds_path = figure_directory / "mod2_thresholds.png"
    plot_pr_curve(validation_metrics, validation_floor, pr_curve_path, "XGBoost")
    plot_thresholds(validation_metrics, thresholds_path)

    # save the model 
    save_model(model_path_raw=MODEL_PATH, metrics=validation_metrics, pipeline=model)

    # weights and bias history
    training_history = model.evals_result()["validation_0"]["logloss"]
    validation_history = model.evals_result()["validation_1"]["logloss"]

    # history expects one dict per round in form -> [{train-loss: x, val-loss: y}, ...]
    history = [{"train_logloss": training_loss, "val_logloss": validation_loss} for training_loss, validation_loss in zip(training_history, validation_history)]

    results = {
      "run": "MOD-2",
      "date": date.today().isoformat(),
      "xgb_version": xgb.__version__,
      "train_rows": len(training),
      "validation_rows": len(validation),
      "features": {"ordinal": ORDINAL_COLUMNS, "numerical": NUMERIC_COLUMNS, "category": CATEGORICAL_COLUMNS},
      "hyperparameters": {
          "tree_method": TREE_METHOD,
          "eval_metric": EVAL_METRIC,
          "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
          "n_estimators": N_ESTIMATORS,
          "random_state": RANDOM_STATE,
          "learning_rate": LEARNING_RATE,
          "max_depth": MAX_DEPTH,
          "min_child_weight": MIN_CHILD_WEIGHT,
          "subsample": SUBSAMPLE,
          "colsample_bytree": COLSAMPLE_BYTREE,
          "reg_lambda": REG_LAMBDA,
          "reg_alpha": REG_ALPHA,
          "gamma": GAMMA,
          "max_cat_to_onehot": MAX_CAT_TO_ONEHOT,
          "max_cat_threshold": MAX_CAT_THRESHOLD,
      },
      "best_iteration": int(model.best_iteration),
      "rounds_trained": len(history),
    }

    results["train"] = training_metrics
    results["validation"] = validation_summary(validation_metrics, validation_floor)

    # save the json
    save_json(METRICS_PATH, results)

    # save to weights and biases
    log_to_wandb(results=results, figures={"pr_curve": pr_curve_path, "thresholds": thresholds_path}, tags=[WANDB_MODEL_TAG, WANDB_FEATURE_TAG], history=history)



if __name__ == "__main__":
    main()



    