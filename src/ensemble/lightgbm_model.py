from datetime import date
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from src.evaluation import evaluate, print_report, plot_pr_curve, plot_thresholds, save_model, validation_summary, save_json, log_to_wandb, compute_loss
import argparse
import time
import json

# Constants

ORDINAL_COLUMNS = ["DepTimeBlk", "ArrTimeBlk"]
CATEGORICAL_COLUMNS = ["Origin", "Dest", "OriginCityMarketID", "DestCityMarketID", "OriginState", "DestState", "Reporting_Airline"]
NUMERIC_COLUMNS = ["CRSElapsedTime", "Distance", "DayofMonth", "DayOfWeek"]
FEATURES = ORDINAL_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET = "ArrDel15"

MODEL_PATH = "models/mod3_lightgbm.joblib"
PREPROCESS_PATH = "data/processed"
FIGURES_PATH = "reports/lightgbm/figures"
METRICS_PATH = "reports/lightgbm/mod3_metrics.json"
WANDB_MODEL_TAG = "lightgbm"
WANDB_FEATURE_TAG = "13-baseline-features"

# training setup
RANDOM_STATE = 42
EVAL_METRIC = "binary_logloss"
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS = 6000    # upper limit; early stopping picks the actual tree count

# hyperparameters (Lightgbm defaults)
LEARNING_RATE = 0.1
NUM_LEAVES = 31
MAX_DEPTH = -1 # no limit on max depth in lightgbm defaults
MIN_CHILD_SAMPLES = 20
SUBSAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0
REG_LAMBDA = 0.0
REG_ALPHA = 0.0
MIN_SPLIT_GAIN = 0.0
MAX_CAT_TO_ONEHOT = 4
MAX_CAT_THRESHOLD = 32
CAT_SMOOTH = 10

FIXED_PARAMS = {"objective": "binary",
                "verbose": -1,
                "subsample_freq": 1,
                "n_estimators":N_ESTIMATORS,
                "random_state":RANDOM_STATE}



DEFAULT_HYPERPARAMS_DICT = {"learning_rate":LEARNING_RATE,
                "num_leaves": NUM_LEAVES,
                "max_depth":MAX_DEPTH,
                "min_child_samples":MIN_CHILD_SAMPLES,
                "subsample":SUBSAMPLE,
                "colsample_bytree":COLSAMPLE_BYTREE,
                "reg_lambda":REG_LAMBDA,
                "reg_alpha":REG_ALPHA,
                "min_split_gain":MIN_SPLIT_GAIN,
                "max_cat_to_onehot":MAX_CAT_TO_ONEHOT,
                "max_cat_threshold":MAX_CAT_THRESHOLD,
                "cat_smooth": CAT_SMOOTH
                }


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

def prepare_data(): 
    # load the splits
    training, validation = load_splits()

    # split the target off the datasets
    training, training_y, validation, validation_y = split_target_from_data(training=training, validation=validation)

    # encode the columns to categories, used by both xgboost and lightgbm
    training, validation = encode(train=training, val=validation)

    return training, training_y, validation, validation_y

def build_history(model):
    # weights and bias history
    training_history = model.evals_result_["train"]["binary_logloss"]
    validation_history = model.evals_result_["val"]["binary_logloss"]

    # history expects one dict per round in form -> [{train-loss: x, val-loss: y}, ...]
    history = [{"train_logloss": training_loss, "val_logloss": validation_loss} for training_loss, validation_loss in zip(training_history, validation_history)]
    return history

def build_results(model, training, training_metrics, validation, validation_metrics, validation_floor, history, train_seconds, hyperparams, params_source=None):
    results = {
      "run": "MOD-3",
      "date": date.today().isoformat(),
      "lgbm_version": lgb.__version__,
      "train_rows": len(training),
      "validation_rows": len(validation),
      "features": {"ordinal": ORDINAL_COLUMNS, "numerical": NUMERIC_COLUMNS, "category": CATEGORICAL_COLUMNS},
      "train_seconds": train_seconds,
      "hyperparameters": {
          **FIXED_PARAMS,
          **hyperparams
      },
        # XGBoost's best_iteration is 0-based, LightGBM's is 1-based; stored 0-based so MOD-2 and MOD-3 compare directly. model.best_iteration_ itself is left alone because predict_proba uses it to pick the number of trees.
      "best_iteration": int(model.best_iteration_ - 1),      
      "rounds_trained": len(history),
    }

    # only for --params runs: which file the settings came from and what --set changed
    if params_source is not None:
        results["params_source"] = params_source

    results["train"] = training_metrics
    results["validation"] = validation_summary(validation_metrics, validation_floor)

    return results

def save_outputs(results, validation_metrics, validation_floor, model):
    # create the figures 
    figure_directory = Path(FIGURES_PATH)
    figure_directory.mkdir(parents=True, exist_ok=True)
    
    pr_curve_path = figure_directory / "mod3_pr_curve.png"
    thresholds_path = figure_directory / "mod3_thresholds.png"
    plot_pr_curve(validation_metrics, validation_floor, pr_curve_path, "LightGBM")
    plot_thresholds(validation_metrics, thresholds_path)
    
    # save the model 
    save_model(model_path_raw=MODEL_PATH, metrics=validation_metrics, pipeline=model)
    
    # save the json
    save_json(METRICS_PATH, results)

    return {
        "pr_curve": pr_curve_path,
        "thresholds": thresholds_path
    }
    

def build_model(fixed_params, hyperparams: dict):
    model = lgb.LGBMClassifier(**fixed_params,
                              **hyperparams)
    return model

def train_and_evaluate(hyperparams, data, tags, save, params_source=None):
    training, training_y, validation, validation_y = data
    
    # make the model
    model = build_model(FIXED_PARAMS, hyperparams)

    # start the timer to time how long it takes this model to train
    start_time = time.perf_counter()
    
    # fit the model, callback -> log_evalation will report every 100 rounds
    model.fit(training, training_y, eval_X=(training, validation), eval_y=(training_y, validation_y), eval_metric=EVAL_METRIC, eval_names=["train", "val"], callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(100)])

    # stop the timer to get the time to train
    train_seconds = round(time.perf_counter() - start_time, 1)

    # create the history
    history = build_history(model=model)
    
    # print where the model stopped and what the log loss at that round
    best_iteration = model.best_iteration_
    best_log_loss = model.best_score_["val"]["binary_logloss"]
    print(f'Best Iteration: {best_iteration - 1}, Best Log Loss: {best_log_loss}, Time to train: {train_seconds}')
    
    # training and validation probabilites
    training_probabilities = model.predict_proba(training)[:, 1]
    validation_probabilities = model.predict_proba(validation)[:, 1]
    validation_floor = validation_y.mean()
    
    validation_metrics = evaluate(validation_y, validation_probabilities)
    training_metrics = compute_loss(training_y, training_probabilities)
    print_report(validation_metrics, validation_floor)
    print(f'Training PR-auc-over-floor: {training_metrics["pr_auc_over_floor"]}')
    
    results = build_results(model=model, 
                            training=training, 
                            training_metrics=training_metrics,
                            validation=validation,
                            validation_metrics=validation_metrics,
                            validation_floor=validation_floor,
                            history=history,
                            train_seconds=train_seconds,
                            hyperparams=hyperparams,
                            params_source=params_source)
    
    # save to weights and biases
    figure_paths = save_outputs(results=results, validation_metrics=validation_metrics, validation_floor=validation_floor, model=model) if save else {}
    log_to_wandb(results=results, figures=figure_paths, tags=tags, history=history)

    return validation_metrics["pr_auc"]

def load_params(path, overrides):
    res = json.loads(Path(path).read_text())
    all_params = res.get("hyperparameters", res)

    params = {hyperparam_label : all_params[hyperparam_label] for hyperparam_label in all_params if hyperparam_label not in FIXED_PARAMS}

    for override in overrides:
        key, value = override.split("=", 1)
        if key not in DEFAULT_HYPERPARAMS_DICT:
            raise ValueError(f"--set {key}: not a tunable hyperparameter, expected one of {sorted(DEFAULT_HYPERPARAMS_DICT)}")
        params[key] = json.loads(value)
    return to_params(params)

def to_params(trial):
    hyperparams = {**DEFAULT_HYPERPARAMS_DICT, **trial}
    hyperparams["num_leaves"] = int(hyperparams["num_leaves"])
    hyperparams["max_cat_threshold"] = int(hyperparams["max_cat_threshold"])
    hyperparams["min_child_samples"] = int(hyperparams["min_child_samples"])
    return hyperparams

def run_hyperopt(data, max_evals):
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.02), np.log(0.3)),
        "min_child_samples": hp.qloguniform("min_child_samples", np.log(20), np.log(2000), 1),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(.1), np.log(20)),
        "num_leaves": hp.qloguniform("num_leaves", np.log(15), np.log(255), 1),
        "max_cat_threshold": hp.quniform("max_cat_threshold", 8, 128, 8),
        "subsample": hp.uniform("subsample", .5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", .5, 1.0)
    }

    def objective(trial):
        hyperparams = to_params(trial)
        pr_auc = train_and_evaluate(hyperparams=hyperparams, data=data, tags=[WANDB_MODEL_TAG, WANDB_FEATURE_TAG, "hyperopt-trial"], save=False)
        return {"loss": -pr_auc, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE)
    )

    final_params = to_params(best)
    train_and_evaluate(final_params, data, [WANDB_MODEL_TAG, WANDB_FEATURE_TAG, "final"], save=True)
    
    

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tune", type=int, default=0)
    mode.add_argument("--params", type=Path, default=None, help="train and save once with the hyperparameters in this JSON file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override one hyperparameter from --params for this run (repeatable)")
    args = parser.parse_args()

    if args.overrides and args.params is None:
        parser.error("--set requires --params")

    data = prepare_data()
    if args.tune > 0:
        run_hyperopt(data, args.tune)
    elif args.params is not None:
        params = load_params(args.params, args.overrides)
        params_source = {"file": str(args.params), "overrides": args.overrides}
        train_and_evaluate(params, data, [WANDB_MODEL_TAG, WANDB_FEATURE_TAG, "final"], save=True, params_source=params_source)
    else:
      train_and_evaluate(DEFAULT_HYPERPARAMS_DICT, data, [WANDB_MODEL_TAG, WANDB_FEATURE_TAG, "default"], save=True)



if __name__ == "__main__":
    main()



    