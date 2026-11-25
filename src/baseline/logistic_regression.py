import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
from datetime import date
import argparse

# importing the evalution metrics and feature functions
from src.evaluation import evaluate, print_report, plot_pr_curve, plot_thresholds, save_model, validation_summary, save_json, log_to_wandb, compute_loss
from src.features.late_rate import add_late_rate_features

# Constants
MODEL_PATH = "models/mod1_logreg.joblib"
PREPROCESS_PATH = "data/processed"
ONE_HOT_COLUMNS = ["Reporting_Airline", "Origin", "Dest", "DepTimeBlk", "ArrTimeBlk", "OriginState", "DestState",
                   "OriginCityMarketID", "DestCityMarketID", "DayofMonth", "DayOfWeek"]
SCALED_COLUMNS = ["CRSElapsedTime", "Distance"]
TARGET = "ArrDel15"

FIGURES_PATH = "reports/logistic_regression/figures"
METRICS_PATH = "reports/logistic_regression/mod1_metrics.json"

# hyperparameters
MAX_ITERS = 1000
# LEARNING_RATE = .001
REGULARIZATION_CONST = 1.0
CONVERGENCE_LIMIT = 1e-4

# weights and biases constants
WANDB_MODEL_TAG = "logistic-regression"
WANDB_FEATURE_TAG = "13-baseline-features"

RUNS = {
    False: {"run": "MOD-1", "feature_tag": WANDB_FEATURE_TAG, "metrics": METRICS_PATH,
            "model": MODEL_PATH, "prefix": "mod1"},
    True:  {"run": "EXP-4-logistic-regression", "feature_tag": "13-plus-late-rate",
            "metrics": "reports/logistic_regression/exp4_metrics.json",
            "model": "models/exp4_logreg.joblib", "prefix": "exp4"},
    }

def load_splits():
    training_split = pd.read_parquet(Path(PREPROCESS_PATH) / "train.parquet")
    validation_split = pd.read_parquet(Path(PREPROCESS_PATH) / "val.parquet")
    return training_split, validation_split

def split_target_from_data(training, validation):
    training_y = training.pop(TARGET)
    validation_y = validation.pop(TARGET)

    return training, training_y, validation, validation_y

def build_pipeline(scaled_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT_COLUMNS),
            ("scaler", StandardScaler(), scaled_cols)
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=MAX_ITERS, C=REGULARIZATION_CONST, tol=CONVERGENCE_LIMIT))
        ]
    )
    return pipeline


def main():
    # adding the argparser 
    parser = argparse.ArgumentParser()
    parser.add_argument("--late-rate", action="store_true", default=False, help="EXP-4: add the historical late-rate feature")
    args = parser.parse_args()

    training, validation = load_splits()
    training, training_y, validation, validation_y = split_target_from_data(training, validation)

    if args.late_rate:
        # add the late rate column to the data
        training["late_rate"], validation["late_rate"] = add_late_rate_features(training=training, training_y=training_y, validation=validation)

    outputs = RUNS[args.late_rate]
    scaled_cols = SCALED_COLUMNS if not args.late_rate else SCALED_COLUMNS + ["late_rate"]

    pipeline = build_pipeline(scaled_cols=scaled_cols)

    # fits the model on our training data
    pipeline.fit(training, training_y)

    # training probabilities
    training_probabilities = pipeline.predict_proba(training)[:,1]

    # give us the raw probabilities so we can find the correct threshold to predict at
    # predict_proba returns two cols -> [p(ontime) [p(late)]]
    validation_probabilities = pipeline.predict_proba(validation)[:,1]
    print(f"Iterations of Logistic regression: {pipeline.named_steps['model'].n_iter_[0]} / {MAX_ITERS}")

    floor = validation_y.mean()
    metrics = evaluate(validation_y, validation_probabilities)
    print_report(metrics, floor)

    figure_directory = Path(FIGURES_PATH)
    figure_directory.mkdir(parents=True, exist_ok=True)

    pr_curve_path = figure_directory / f"{outputs['prefix']}_pr_curve.png"
    thresholds_path = figure_directory / f"{outputs['prefix']}_thresholds.png"
    plot_pr_curve(metrics, floor, pr_curve_path, "Logistic Regression")
    plot_thresholds(metrics, thresholds_path)

    # save the model
    save_model(model_path_raw=outputs['model'], metrics=metrics, pipeline=pipeline)

    # results must be manually created per model since hyperparameters, scaled cols, one hot cols change per model
    results = {
        "run": outputs["run"],
        "date": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "features": {"one_hot": ONE_HOT_COLUMNS, "scaled": scaled_cols},
        "hyperparameters": {"C": REGULARIZATION_CONST, "tol": CONVERGENCE_LIMIT, "max_iter": MAX_ITERS},
        "iterations": int(pipeline.named_steps["model"].n_iter_[0]),
    }

    # add the validation results such as F1 score, PR-auc, confusion matrix since this can reused per model
    results["validation"] = validation_summary(metrics=metrics, floor=floor)
    results["train"] = compute_loss(training_y, training_probabilities)

    # save the metrics to the specified path with the results
    save_json(outputs["metrics"], results=results)

    # save the metrics to weights and biases
    log_to_wandb(results=results, figures={"pr_curve": pr_curve_path, "thresholds": thresholds_path}, tags=[WANDB_MODEL_TAG, outputs["feature_tag"]])

if __name__ == "__main__":
    main()



    






    