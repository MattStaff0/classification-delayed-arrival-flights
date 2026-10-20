import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import sklearn
from datetime import date

# Constants
MODEL_PATH = "models/mod1_logreg.joblib"
PREPROCESS_PATH = "data/processed"
ONE_HOT_COLUMNS = ["Reporting_Airline", "Origin", "Dest", "DepTimeBlk", "ArrTimeBlk", "OriginState", "DestState",
                   "OriginCityMarketID", "DestCityMarketID", "DayofMonth", "DayOfWeek"]
SCALED_COLUMNS = ["CRSElapsedTime", "Distance"]
TARGET = "ArrDel15"

FIGURES_PATH = "reports/logistic_regression/figures"
METRICS_PATH = "reports/logistic_regression/mod1_metrics.json"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"

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

# Evaluation 
def evaluate(y_true, probabilities):
    pr_auc = average_precision_score(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    # drop the extra trailing point on the recall and precision series
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-12)  # esp term so we do not divide by 0

    # best threshold 
    best_thresh = np.argmax(f1)
    threshold = thresholds[best_thresh]

    # recompute at the given threshold
    preds = (probabilities >= threshold).astype(int)

    # get the confusion matrix counts
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    # precision, recall, f1 score, no support val
    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="binary")

    return {"pr_auc": pr_auc,
            "threshold": threshold,
            "precision": p,
            "recall": r,
            "f1": f,
            "curve": (precision, recall, thresholds, f1),
            "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}}


# Report
def print_report(m ,floor):
    # floor for the pr-auc is just the late rate of the validation set
    print(f"PR-AUC: {m['pr_auc']:.3f} (Floor is validation late rate (total late flights / total flights): {floor:.3f})")
    print(f"threshold: {m['threshold']:.3f}")
    print(f"precision: {m['precision']:.3f}")
    print(f"recall: {m['recall']:.3f}")
    print(f"F1 {m['f1']:.3f}")

# Plotting
def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e0", linewidth=.8)
    ax.set_axisbelow(True)

def plot_pr_curve(m, floor, path):
    precision, recall, _, _ = m["curve"]
    figure, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(recall, precision, color=BLUE, linewidth=2, label=f"Logistic Regression (PR-AUC {m['pr_auc']:.3f})")

    ax.axhline(floor, color=GRAY, linewidth=1, linestyle="--", label=f"No-Skill Floor ({floor:.3f})")

    ax.scatter(m["recall"], m["precision"], color=BLUE, s=64, zorder=3, edgecolor="white", linewidth=2)

    ax.annotate(f"Threshold {m["threshold"]:.3f}", (m["recall"], m["precision"]), xytext=(8,8), textcoords="offset points")
   
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1))
    ax.legend(frameon=False)

    style_axes(ax)
    figure.savefig(path, dpi=150, bbox_inches="tight")

    plt.close(figure)

def plot_thresholds(m, path):
    precision, recall, thresholds, f1 = m["curve"]
    figure, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(thresholds, precision, color=BLUE, linewidth=2, label="Precision")
    ax.plot(thresholds, recall, color=ORANGE, linewidth=2, label="Recall")
    ax.plot(thresholds, f1, color=AQUA, linewidth=2, label="F1 Score")

    ax.axvline(m["threshold"], color=GRAY, linewidth=1, linestyle="--")
    ax.annotate(f"max F1 at {m['threshold']:.2f}", (m["threshold"], 0.95), xytext=(6, 0), textcoords="offset points")

    ax.set(xlabel="Threshold", ylabel="score", ylim=(0,1), title="Choosing the correct threshold on validation")

    ax.legend(frameon=False)
    style_axes(ax)

    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


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
    plot_pr_curve(metrics, floor, figure_directory / "mod1_pr_curve.png")
    plot_thresholds(metrics, figure_directory / "mod1_thresholds.png")

    # save the model
    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": metrics["threshold"]}, model_path)
    print(f"Saved model to {model_path}")

    # save the metrics for this run (numpy values converted to plain python so json can write them)
    results = {
        "run": "MOD-1",
        "date": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "features": {"one_hot": ONE_HOT_COLUMNS, "scaled": SCALED_COLUMNS},
        "hyperparameters": {"C": REGULARIZATION_CONST, "tol": CONVERGENCE_LIMIT, "max_iter": MAX_ITERS},
        "iterations": int(pipeline.named_steps["model"].n_iter_[0]),
        "validation": {
            "pr_auc": float(metrics["pr_auc"]),
            "floor": float(floor),
            "f1_floor": float(2 * floor / (1 + floor)),
            "lift": float(metrics["pr_auc"] / floor),       # lift is how much better the model is how many times better the model is than guessing, needed since we have to compre this model once we tune it against the testing data
            "threshold": float(metrics["threshold"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "confusion_matrix": metrics["confusion"],
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()



    






    