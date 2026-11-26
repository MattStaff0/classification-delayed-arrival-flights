from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import wandb

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
WANDB_PROJECT = "flight-classifier"
WANDB_ENTITY = "mattstaff0-texas-christian-university"


# Evaluation 
def evaluate(y_true, probabilities):
    pr_auc = average_precision_score(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    logloss = log_loss(y_true, probabilities)

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
            "log_loss": logloss,
            "threshold": threshold,
            "precision": p,
            "recall": r,
            "f1": f,
            "curve": (precision, recall, thresholds, f1),
            "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}}

def compute_loss(y_true, probabilities):
    floor = y_true.mean()
    pr_auc = average_precision_score(y_true, probabilities)

    pr_auc_over_floor = pr_auc / floor
    logloss = log_loss(y_true, probabilities)

    return {
        "log_loss": float(logloss),
        "pr_auc": float(pr_auc),
        "floor": float(floor),
        "pr_auc_over_floor": float(pr_auc_over_floor)
    }

# EVAL-4: score a split at a threshold that was chosen on the validation set
def score_at_threshold(y_true, probs, threshold):
    # scores that do not require thresholds
    scores = compute_loss(y_true, probs)

    # label each of the flights with the threshold chosen from validation for our predictions
    preds = (probs >= threshold).astype(int)

    # get the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()

    # get the precision, recall, f1 score, zero divison makes sure no errors if we divide by 0
    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)

    # get the floor for the testing set
    floor = scores["floor"]
    scores.update({
        "f1_floor": float(2 * floor / (1 + floor)), # f1 if we predicted everything as late
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    })
    return scores


# Report
def print_report(m ,floor):
    # floor for the pr-auc is just the late rate of the validation set
    print("For the validation set:")
    print(f"PR-AUC: {m['pr_auc']:.3f} (Floor is validation late rate (total late flights / total flights): {floor:.3f})")
    print(f"PR-AUC-over-floor: {float(m["pr_auc"] / floor)}")
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
                "log_loss": float(metrics["log_loss"]),
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

# connection to weights and biases
def log_to_wandb(results, figures, tags, history=None):
    """
    Log the results and figures to Weights and Biases (wandb), both training and validation metrics are logged to the run summary. If a history of metrics is provided, it will be logged as well.

    Args:
        results (dict): Dictionary containing the results to log.
        figures (dict): Dictionary containing the figures to log.
        tags (list): List of tags to associate with the wandb run.
        history (list, optional): List of dictionaries containing the history of metrics to log. Defaults to None.
    
    Returns:
        None
    """
    # initialize a new run
    with wandb.init(name=results["run"], 
                     project=WANDB_PROJECT, 
                     entity=WANDB_ENTITY, 
                     tags=tags, 
                     config={label: results[label] for label in results.keys() if label not in ["validation", "train"]}) as run:

        # log both the training and validation metrics to the run summary
        overall_metrics = {}
        overall_metrics["validation"] = results["validation"]
        overall_metrics["train"] = results["train"]

        run.summary.update(overall_metrics)

        # log the history if it exists
        if history:
            for i in range(len(history)):
                run.log(history[i], step=i)
                
        # log the figures to the run
        run.log({figure_name: wandb.Image(figures[figure_name]) for figure_name in figures.keys()})




