"""
Export W&B runs (e.g. every hyperopt trial for one model) to a CSV in reports/.

The metrics JSON only keeps the final model; this keeps a committed record of every
configuration tried.

Run from the project root:
    uv run python scripts/export_wandb_runs.py --run MOD-2 --tag hyperopt-trial --out reports/xgboost/mod2_trials.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import wandb

# use the same project and team as log_to_wandb
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import WANDB_ENTITY, WANDB_PROJECT

# summary metrics to keep (W&B stores the nested blocks with dots in the key)
SUMMARY_KEYS = ["validation.pr_auc", "validation.pr_auc_over_floor", "validation.f1", "validation.threshold",
                "validation.precision", "validation.recall", "validation.log_loss",
                "train.pr_auc", "train.pr_auc_over_floor", "train.log_loss"]


def get_runs(run_name, tag):
    # longer timeout: fetching ~100 runs can take a while
    api = wandb.Api(timeout=120)
    # let W&B filter by run name and tag, instead of downloading every run in the project
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                    filters={"display_name": run_name, "tags": {"$in": [tag]}},
                    per_page=100)
    # oldest first so row 1 is trial 1
    return sorted(runs, key=lambda run: run.created_at)


def run_to_row(trial_number, run):
    row = {"trial": trial_number, "run_id": run.id}

    # every hyperparameter the run used
    for name, value in run.config["hyperparameters"].items():
        row[name] = value

    row["best_iteration"] = run.config["best_iteration"]
    row["rounds_trained"] = run.config["rounds_trained"]

    for key in SUMMARY_KEYS:
        row[key] = run.summary[key]

    # overfitting gap: how much better the model is on train than on validation
    row["gap_pct"] = 100 * (row["train.pr_auc_over_floor"] / row["validation.pr_auc_over_floor"] - 1)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="MOD-2")
    parser.add_argument("--tag", default="hyperopt-trial")
    parser.add_argument("--out", default="reports/xgboost/mod2_trials.csv")
    args = parser.parse_args()

    runs = get_runs(args.run, args.tag)
    table = pd.DataFrame([run_to_row(i + 1, run) for i, run in enumerate(runs)])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"Saved {len(table)} runs to {out_path}")


if __name__ == "__main__":
    main()
