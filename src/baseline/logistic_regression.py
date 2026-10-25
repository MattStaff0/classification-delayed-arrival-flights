import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
from datetime import date

# importing the evalution metrics file
from src.evaluation import evaluate, print_report, plot_pr_curve, plot_thresholds, save_model, validation_summary, save_json


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

def load_splits():
    training_split = pd.read_parquet(Path(PREPROCESS_PATH) / "train.parquet")
    validation_split = pd.read_parquet(Path(PREPROCESS_PATH) / "val.parquet")
    return training_split, validation_split

def split_target_from_data(training, validation):
    training_y = training.pop(TARGET)
    validation_y = validation.pop(TARGET)

    return training, training_y, validation, validation_y

def build_pipeline():
    # return the unfitted pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT_COLUMNS),
            ("scaler", StandardScaler(), SCALED_COLUMNS)
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
    training, validation = load_splits()
    training, training_y, validation, validation_y = split_target_from_data(training, validation)

    pipeline = build_pipeline()

    # fits the model on our training data
    pipeline.fit(training, training_y)

    # give us the raw probabilities so we can find the correct threshold to predict at
    # predict_proba returns two cols -> [p(ontime) [p(late)]]
    validation_probabilities = pipeline.predict_proba(validation)[:,1]
    print(f"Iterations of Logistic regression: {pipeline.named_steps['model'].n_iter_[0]} / {MAX_ITERS}")

    floor = validation_y.mean()
    metrics = evaluate(validation_y, validation_probabilities)
    print_report(metrics, floor)

    figure_directory = Path(FIGURES_PATH)
    figure_directory.mkdir(parents=True, exist_ok=True)
    plot_pr_curve(metrics, floor, figure_directory / "mod1_pr_curve.png", "Logistic Regression")
    plot_thresholds(metrics, figure_directory / "mod1_thresholds.png")

    # save the model
    save_model(model_path_raw=MODEL_PATH, metrics=metrics, pipeline=pipeline)

    # results must be manually created per model since hyperparameters, scaled cols, one hot cols change per model
    results = {
        "run": "MOD-1",
        "date": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "features": {"one_hot": ONE_HOT_COLUMNS, "scaled": SCALED_COLUMNS},
        "hyperparameters": {"C": REGULARIZATION_CONST, "tol": CONVERGENCE_LIMIT, "max_iter": MAX_ITERS},
        "iterations": int(pipeline.named_steps["model"].n_iter_[0]),
    }

    # add the validation results such as F1 score, PR-auc, confusion matrix since this can reused per model
    results["validation"] = validation_summary(metrics=metrics, floor=floor)

    # save the metrics to the specified path with the results
    save_json(METRICS_PATH, results=results)

if __name__ == "__main__":
    main()



    






    