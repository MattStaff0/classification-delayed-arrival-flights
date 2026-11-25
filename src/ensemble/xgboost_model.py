from datetime import date
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from src.evaluation import evaluate, print_report, plot_pr_curve, plot_thresholds, save_model, validation_summary, save_json, log_to_wandb, compute_loss
from src.features.late_rate import add_late_rate_features
import argparse
import time
import json

# Constants

ORDINAL_COLUMNS = ["DepTimeBlk", "ArrTimeBlk"]
CATEGORICAL_COLUMNS = ["Origin", "Dest", "OriginCityMarketID", "DestCityMarketID", "OriginState", "DestState", "Reporting_Airline"]
NUMERIC_COLUMNS = ["CRSElapsedTime", "Distance", "DayofMonth", "DayOfWeek"]
FEATURES = ORDINAL_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET = "ArrDel15"

MODEL_PATH = "models/mod2_xgboost.joblib"
FIGURES_PATH = "reports/xgboost/figures"
METRICS_PATH = "reports/xgboost/mod2_metrics.json"
WANDB_FEATURE_TAG = "13-baseline-features"
WANDB_MODEL_TAG = 'xgboost'

# names here must match preprocess.py's datasets
DATA_FOLDERS = {"base": "data/processed", "fullyear": "data/processed_fullyear"}

# late_rate, dataset, run name, feature tag, data tag, and output files for each kind of run, so a run never overwrites another runs files
RUNS = {
    "MOD-2": {"late_rate": False, "dataset": "base", "feature_tag": WANDB_FEATURE_TAG, "data_tag": None,
            "metrics": METRICS_PATH, "model": MODEL_PATH, "prefix": "mod2"},
    "EXP-4-xgboost": {"late_rate": True, "dataset": "base", "feature_tag": "13-plus-late-rate", "data_tag": None,
            "metrics": "reports/xgboost/exp4_metrics.json", "model": "models/exp4_xgboost.joblib", "prefix": "exp4"},
    "EXP-2": {"late_rate": False, "dataset": "fullyear", "feature_tag": WANDB_FEATURE_TAG, "data_tag": "21-training-months",
            "metrics": "reports/xgboost/exp2_metrics.json", "model": "models/exp2_xgboost.joblib", "prefix": "exp2"},
}

# training setup
RANDOM_STATE = 42
TREE_METHOD = "hist"
EVAL_METRIC = "logloss"
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS = 6000    # upper limit; early stopping picks the actual tree count

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

FIXED_PARAMS = {"tree_method":TREE_METHOD,
                "enable_categorical":True,
                "eval_metric":EVAL_METRIC,
                "early_stopping_rounds":EARLY_STOPPING_ROUNDS,
                "n_estimators":N_ESTIMATORS,
                "random_state":RANDOM_STATE}



DEFAULT_HYPERPARAMS_DICT = {"learning_rate":LEARNING_RATE,
                "max_depth":MAX_DEPTH,
                "min_child_weight":MIN_CHILD_WEIGHT,
                "subsample":SUBSAMPLE,
                "colsample_bytree":COLSAMPLE_BYTREE,
                "reg_lambda":REG_LAMBDA,
                "reg_alpha":REG_ALPHA,
                "gamma":GAMMA,
                "max_cat_to_onehot":MAX_CAT_TO_ONEHOT,
                "max_cat_threshold":MAX_CAT_THRESHOLD}


def load_splits(folder):
    training_split = pd.read_parquet(Path(folder) / "train.parquet")
    validation_split = pd.read_parquet(Path(folder) / "val.parquet")
    return training_split, validation_split

def split_target_from_data(training, validation):
    training_y = training.pop(TARGET)
    validation_y = validation.pop(TARGET)
    return training, training_y, validation, validation_y

def encode(train, val, features):
    # keep only the model's features (13 base, plus late_rate for EXP-4)
    train = train[features]
    val = val[features]

    # make the marketID's into categories
    train["OriginCityMarketID"] = train["OriginCityMarketID"].astype("category")
    train["DestCityMarketID"] = train["DestCityMarketID"].astype("category")

    for col in CATEGORICAL_COLUMNS + ORDINAL_COLUMNS:
        val[col] = pd.Categorical(val[col], categories=train[col].cat.categories)

    return train, val

def prepare_data(late_rate=False, dataset="base"): 
    # load the splits
    training, validation = load_splits(DATA_FOLDERS[dataset])

    # split the target off the datasets
    training, training_y, validation, validation_y = split_target_from_data(training=training, validation=validation)

    # the late rate experiment the column needs to be added
    if late_rate:
        training["late_rate"], validation["late_rate"] = add_late_rate_features(training=training, training_y=training_y, validation=validation)

    # add the late rate feature if late_rate is true else just pass in the 13 normal features
    features = FEATURES if not late_rate else FEATURES + ["late_rate"]

    # encode the columns to categories for xgboost
    training, validation = encode(train=training, val=validation, features=features)

    return training, training_y, validation, validation_y

def run_name(late_rate, dataset):
    # flags decide what run this is, not the data
    for name, run in RUNS.items():
        if run["late_rate"] == late_rate and run["dataset"] == dataset:
            return name
    # if we ran the late rate with the full year of data, not configured yet
    return None

def build_history(model):
    # weights and bias history
    training_history = model.evals_result()["validation_0"]["logloss"]
    validation_history = model.evals_result()["validation_1"]["logloss"]

    # history expects one dict per round in form -> [{train-loss: x, val-loss: y}, ...]
    history = [{"train_logloss": training_loss, "val_logloss": validation_loss} for training_loss, validation_loss in zip(training_history, validation_history)]
    return history

def build_results(model, run, training, training_metrics, validation, validation_metrics, validation_floor, history, train_seconds, hyperparams, params_source=None):
    # record what the model actually trained on, read from the data itself
    numeric_cols = NUMERIC_COLUMNS + ["late_rate"] if "late_rate" in training.columns else NUMERIC_COLUMNS
    results = {
      "run": run,
      "date": date.today().isoformat(),
      "xgb_version": xgb.__version__,
      "train_rows": len(training),
      "validation_rows": len(validation),
      "features": {"ordinal": ORDINAL_COLUMNS, "numerical": numeric_cols, "category": CATEGORICAL_COLUMNS},
      "train_seconds": train_seconds,
      "hyperparameters": {
          **FIXED_PARAMS,
          **hyperparams
      },
      "best_iteration": int(model.best_iteration),
      "rounds_trained": len(history),
    }

    # only for --params runs: which file the settings came from and what --set changed
    if params_source is not None:
        results["params_source"] = params_source

    # only for runs not on the base data (EXP-2), which dataset folder train came from, so other metric files stay as they are
    dataset = RUNS[run]["dataset"]
    if dataset != "base":
        results["dataset"] = {"name": dataset, "folder": DATA_FOLDERS[dataset]}

    results["train"] = training_metrics
    results["validation"] = validation_summary(validation_metrics, validation_floor)

    return results

def save_outputs(results, validation_metrics, validation_floor, model):
    # file names for this kind of run (MOD-2 or EXP-4)
    outputs = RUNS[results["run"]]

    # create the figures
    figure_directory = Path(FIGURES_PATH)
    figure_directory.mkdir(parents=True, exist_ok=True)

    pr_curve_path = figure_directory / f"{outputs['prefix']}_pr_curve.png"
    thresholds_path = figure_directory / f"{outputs['prefix']}_thresholds.png"
    plot_pr_curve(validation_metrics, validation_floor, pr_curve_path, "XGBoost")
    plot_thresholds(validation_metrics, thresholds_path)

    # save the model
    save_model(model_path_raw=outputs["model"], metrics=validation_metrics, pipeline=model)

    # save the json
    save_json(outputs["metrics"], results)

    return {
        "pr_curve": pr_curve_path,
        "thresholds": thresholds_path
    }
    

def build_model(fixed_params, hyperparams: dict):
    model = xgb.XGBClassifier(**fixed_params,
                              **hyperparams)
    return model

def train_and_evaluate(hyperparams, data, run, tags, save, params_source=None):
    training, training_y, validation, validation_y = data
    
    # make the model
    model = build_model(FIXED_PARAMS, hyperparams)

    # start a timer for the training of the model only 
    time_start = time.perf_counter()
    
    # fit the model, verbose will report every 100 rounds the rounds are based on N_ESTIMATORS
    model.fit(training, training_y, eval_set=[(training, training_y), (validation, validation_y)], verbose=100)

    # end the timer as we only care about the timing of training
    train_seconds = round(time.perf_counter() - time_start, 1)
    
    # create the history
    history = build_history(model=model)
    
    # print where the model stopped and what the log loss at that round
    best_iteration = model.best_iteration
    best_log_loss = model.best_score
    print(f'Best Iteration: {best_iteration}, Best Log Loss: {best_log_loss}, Training Seconds: {train_seconds}')
    
    # training and validation probabilites
    training_probabilities = model.predict_proba(training)[:, 1]
    validation_probabilities = model.predict_proba(validation)[:, 1]
    validation_floor = validation_y.mean()
    
    validation_metrics = evaluate(validation_y, validation_probabilities)
    training_metrics = compute_loss(training_y, training_probabilities)
    print_report(validation_metrics, validation_floor)
    print(f'Training PR-auc-over-floor: {training_metrics["pr_auc_over_floor"]}')
    
    results = build_results(model=model, 
                            run=run,
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
    # metrics JSON (uses its "hyperparameters" block) or a plain dict of hyperparameters
    res = json.loads(Path(path).read_text())
    all_params = res.get("hyperparameters", res)

    # fixed params come from the code only; build_model already passes FIXED_PARAMS
    params = {hyperparam_label : all_params[hyperparam_label] for hyperparam_label in all_params if hyperparam_label not in FIXED_PARAMS}

    # --set KEY=VALUE overrides for this run only
    for override in overrides:
        key, value = override.split("=", 1)
        if key not in DEFAULT_HYPERPARAMS_DICT:
            raise ValueError(f"--set {key}: not a tunable hyperparameter, expected one of {sorted(DEFAULT_HYPERPARAMS_DICT)}")
        params[key] = json.loads(value)

    return to_params(params)

def to_params(trial):
    hyperparams = {**DEFAULT_HYPERPARAMS_DICT, **trial}
    hyperparams["max_depth"] = int(hyperparams["max_depth"])
    hyperparams["max_cat_threshold"] = int(hyperparams["max_cat_threshold"])
    return hyperparams

def run_hyperopt(data, max_evals, run):
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.02), np.log(0.3)),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(1), np.log(200)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(.1), np.log(20)),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "max_cat_threshold": hp.quniform("max_cat_threshold", 8, 128, 8),
        "subsample": hp.uniform("subsample", .5 , 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", .5, 1.0)
    }

    def objective(trial):
        hyperparams = to_params(trial)
        pr_auc = train_and_evaluate(hyperparams=hyperparams, data=data, run=run, tags=[WANDB_MODEL_TAG, WANDB_FEATURE_TAG, "hyperopt-trial"], save=False)
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
    train_and_evaluate(final_params, data, run, [WANDB_MODEL_TAG, WANDB_FEATURE_TAG, "final"], save=True)
    
    

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tune", type=int, default=0)
    mode.add_argument("--params", type=Path, default=None, help="train and save once with the hyperparameters in this JSON file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override one hyperparameter from --params for this run (repeatable)")
    parser.add_argument("--late-rate", action="store_true", default=False, help="EXP-4: add the historical late-rate feature (requires --params so the tuned settings stay fixed)")
    parser.add_argument("--dataset", choices=sorted(DATA_FOLDERS), default="base", help="EXP-2: which processed folder to train on (default: base -> data/processed; anything else requires --params)")
    args = parser.parse_args()

    if args.overrides and args.params is None:
        parser.error("--set requires --params")

    # EXP-4 changes only the feature: no default-settings run, no re-tuning
    if args.late_rate and args.params is None:
        parser.error("--late-rate requires --params")

    # EXP-2 changes only the training data
    if args.dataset != "base" and args.params is None:
        parser.error("--dataset other than base requires --params")

    # flags pick what run we are doing
    run = run_name(args.late_rate, args.dataset)

    if run is None:
        parser.error(f'No run defined for --late-rate:{args.late_rate} with --dataset {args.dataset}: add a row to RUNS and try again.')

    data = prepare_data(late_rate=args.late_rate, dataset=args.dataset)
    if args.tune > 0:
        run_hyperopt(data, args.tune, run)
    elif args.params is not None:
        params = load_params(args.params, args.overrides)
        params_source = {"file": str(args.params), "overrides": args.overrides}

        data_tag = RUNS[run]['data_tag']
        tags = [WANDB_MODEL_TAG, RUNS[run]["feature_tag"]] + ([data_tag] if data_tag else []) + ["final"]
        train_and_evaluate(params, data, run, tags, save=True, params_source=params_source)
    else:
      train_and_evaluate(DEFAULT_HYPERPARAMS_DICT, data, run, [WANDB_MODEL_TAG, WANDB_FEATURE_TAG, "default"], save=True)



if __name__ == "__main__":
    main()



    