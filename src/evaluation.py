from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"


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

def plot_pr_curve(m, floor, path, model_name):
    precision, recall, _, _ = m["curve"]
    figure, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(recall, precision, color=BLUE, linewidth=2, label=f"{model_name} (PR-AUC {m['pr_auc']:.3f})")
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

# Model Saving
def save_model(model_path_raw, pipeline, metrics):
    # save the model
    model_path = Path(model_path_raw)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": metrics["threshold"]}, model_path)
    print(f"Saved model to {model_path}")

# Metric saving
def validation_summary(metrics, floor):
    return {
                "pr_auc": float(metrics["pr_auc"]),
                "floor": float(floor),
                "f1_floor": float(2 * floor / (1 + floor)),
                "pr_auc_over_floor": float(metrics["pr_auc"] / floor),       # pr_auc_over_floor is how much better the model is how many times better the model is than guessing, needed since we have to compare this model once we tune it against the testing data
                "threshold": float(metrics["threshold"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "confusion_matrix": metrics["confusion"],
            }

def save_json(metric_path, results):
    metric_path = Path(metric_path)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metric_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {metric_path}")