import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datetime import date
import lightgbm
import matplotlib.pyplot as plt
import sklearn
import xgboost
from sklearn.metrics import precision_recall_curve
from src.evaluation import score_at_threshold, save_json, log_to_wandb, style_axes, BLUE, ORANGE, AQUA, GRAY

#  EVAL-4: the one place the test split is scored. Imports nothing from the model scripts, so tuning code can never reach it.

DATA_FOLDER = "data/processed"
TARGET = 'ArrDel15'
OUTPUT_PATH = "reports/final/eval4_metrics.json"
FIGURES_PATH = 'reports/final/figures'

WANDB_RUN = "EVAL-4"
WANDB_FEATURE_TAG = "13-baseline-features"
MODEL_COLORS = {"MOD-1": ORANGE, "MOD-2": BLUE, "MOD-3": AQUA}

FINAL_MODELS = {
      "MOD-1": {"model": "models/mod1_logreg.joblib",
            "sha256": "41c81efbbb375d5fc297175fd1f7e07c48ac7ad55425849a77c3b3036b19cc69",
            "metrics": "reports/logistic_regression/mod1_metrics.json",
            "role": "final", "prep": "raw", "model_tag": "logistic-regression"},
      "MOD-2": {"model": "models/mod2_xgboost.joblib",
            "sha256": "bf3157a0a43da10228c282acfc65684ca3044e792170846310bf5c24e574fe67",
            "metrics": "reports/xgboost/mod2_metrics.json",
            "role": "final", "prep": "xgboost", "model_tag": "xgboost"},
      # reference row only: its test result does not change which tree model is final
      "MOD-3": {"model": "models/mod3_lightgbm.joblib",
            "sha256": "bc243388243b5eff20bd5d9076be741a721312aef6579315a5a498ea4fd0d892",
            "metrics": "reports/lightgbm/mod3_metrics.json",
            "role": "reference", "prep": "lightgbm", "model_tag": "lightgbm"},
}

def file_hash(path):
    # load the file in small chunks so a large model file is not loaded all into memory at once
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()

def check_frozen(name, row):
    # stop before scoring if the model file is not the frozen one
    actual = file_hash(row["model"])
    if actual != row['sha256']:
        raise SystemExit(f"{name}: {row['model']} changed since the freeze. (sha256 {actual[:8]} vs frozen {row['sha256'][:8]}). Not scoring it.")

def load_split(split):
    # dry run -> validation and real run -> test
    data = pd.read_parquet(Path(DATA_FOLDER) / f"{split}.parquet")
    y = data.pop(TARGET)
    return data, y

def model_features(model, prep):
      # the columns and order the model was trained with, read from the model itself (Q7)
      if prep == "xgboost":
          return model.get_booster().feature_names
      return model.feature_name_  # lightgbm


def stored_categories(model, prep):
    # {column: category list} saved inside the model at training time, so test is coded exactly like train
    if prep == "xgboost":
        stored = model.get_booster().get_categories(export_to_arrow=True).to_arrow()
        # numeric columns come back as None, only the 9 categorical columns have a list
        return {col: values.to_pylist() for col, values in stored if values is not None}

    if prep == "lightgbm":
        booster = model.booster_
        # pandas_categorical holds the 9 lists but no column names; the model text has the line
        # "[categorical_feature: 0,1,...]" giving their positions in feature_name()
        # num_iteration=1 so it writes out one tree instead of all 2,494
        line = next(l for l in booster.model_to_string(num_iteration=1).splitlines()
                    if l.startswith("[categorical_feature:"))
        positions = [int(i) for i in line.split(":")[1].strip(" ]").split(",")]
        return {booster.feature_name()[i]: cats for i, cats in zip(positions, booster.pandas_categorical)}

    return {}

def prepare(model, data, prep):
    # MOD-1's pipeline one-hot encodes the raw columns itself (an unseen value -> all-zero columns), so pass it through
    if prep == "raw":
          one_hot = model.named_steps["prep"].named_transformers_["one_hot"]
          unseen = {col: int((~data[col].isin(cats)).sum()) for col, cats in zip(one_hot.feature_names_in_, one_hot.categories_)}
          return data, unseen

    prepared = data[model_features(model, prep)].copy()
    unseen = {}
    for col, categories in stored_categories(model, prep).items():
        # position of each value in the stored list, -1 if it is not in the list
        codes = pd.Index(categories).get_indexer(prepared[col])
        unseen[col] = int((codes == -1).sum())
        # from_codes turns -1 into missing directly, so pandas never sees a value outside the list (no Pandas4Warning)
        prepared[col] = pd.Categorical.from_codes(codes, categories=categories)
    return prepared, unseen

def listed_but_untrained(model, prep, data):
    # values in the model's stored list but with 0 training rows -> CSG/LNS, preprocess.py built these lists from the 12 months
    # so a test only value can be listed from here, prepare keeps them as a real category, so unseen does not count them
    # train.parquet used for reading the count 
    stored = stored_categories(model, prep)
    train = pd.read_parquet(Path(DATA_FOLDER) / "train.parquet", columns=list(stored))

    found = {}
    for col, cats in stored.items():
        trained_on = set(train[col].unique())
        values = [value for value in cats if value not in trained_on]
        if values:
            # what values were found and how many rows the scored split carries
            found[col] = {"values": values, "rows": int(data[col].isin(values).sum())}
    return found

def score_model(name, row, data, y):
      # load the frozen model and its validation-chosen threshold, encode from the model, score at that threshold
      saved = joblib.load(row["model"])
      model, threshold = saved["pipeline"], saved["threshold"]

      prepared, unseen = prepare(model, data, row["prep"])
      probabilities = model.predict_proba(prepared)[:, 1]

      result = {
          "model": name,
          "role": row["role"],
          "model_file": row["model"],
          "sha256": row["sha256"],
          "threshold": float(threshold),
          # the numbers frozen on validation, so the scored split sits next to them
          "validation": json.load(open(row["metrics"]))["validation"],
          "scores": score_at_threshold(y, probabilities, threshold),
          "unseen": unseen,
          "listed_but_untrained": listed_but_untrained(model, row["prep"], data) if row["prep"] != "raw" else {},
      }
      return result, probabilities


def dry_run_check(name, result):
    # --split val: validation scored through this script must reproduce the saved metrics json exactly,
    # which proves the loading, encoding, and scoring path before test is ever read

    saved, scored = result["validation"], result["scores"]
    keys = ["pr_auc", "log_loss", "floor", "f1_floor", "pr_auc_over_floor", "threshold", "precision", "recall", "f1"]
    different = [key for key in keys if not np.isclose(scored[key], saved[key], rtol=0, atol=1e-12)]

    if different or scored["confusion_matrix"] != saved["confusion_matrix"]:
        raise SystemExit(f"{name}: dry run does not match {FINAL_MODELS[name]['metrics']} on {different or 'confusion_matrix'}")
    
    print(f"{name}: dry run matches saved validation (PR-AUC {scored['pr_auc']:.4f}, F1 {scored['f1']:.4f}, threshold{scored['threshold']:.4f})")

def test_block(result):
      # score_model names the scored split "scores"; on the real run it is test, and log_to_wandb's SPLITS
      # only sends a block to the summary if it is called "test" (otherwise it would land in config)
      block = {key: value for key, value in result.items() if key != "scores"}
      block["test"] = result["scores"]
      return block


def plot_test_pr_curves(results, probabilities, y, path):
    # all three test PR curves on one plot, each dot is where its frozen validation threshold lands on test
    figure, ax = plt.subplots(figsize=(6, 4.5))
    for name, block in results.items():
        precision, recall, _ = precision_recall_curve(y, probabilities[name])
        test = block["test"]
        style = "--" if block["role"] == "reference" else "-"   # MOD-3 dashed: reference only
        ax.plot(recall, precision, color=MODEL_COLORS[name], linewidth=2, linestyle=style,
                label=f"{name} {block['role']} (PR-AUC {test['pr_auc']:.3f})")
        ax.scatter(test["recall"], test["precision"], color=MODEL_COLORS[name], s=64, zorder=3, edgecolor="white", linewidth=2)

    # every model is scored on the same rows, so the floor (test late rate) is the same for all three
    floor = next(iter(results.values()))["test"]["floor"]
    ax.axhline(floor, color=GRAY, linewidth=1, linestyle=":", label=f"No-skill floor ({floor:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Test (May-Jun 2026), dots at the frozen thresholds")
    ax.legend(frameon=False)
    style_axes(ax)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_val_vs_test(results, path):
    # pr_auc_over_floor, not raw PR-AUC, because validation and test have different late rates (Q12)
    names = list(results)
    x = np.arange(len(names))
    width = 0.38
    figure, ax = plt.subplots(figsize=(6, 4.5))
    for offset, split, color, label in [(-width / 2, "validation", GRAY, "Validation (Apr 2026)"), (width / 2, "test", BLUE, "Test (May-Jun 2026)")]:
        values = [results[name][split]["pr_auc_over_floor"] for name in names]
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=2)

    ax.axhline(1, color=GRAY, linewidth=1, linestyle=":", label="No skill (1.0)")
    ax.set(xticks=x, xticklabels=[f"{name}\n{results[name]['role']}" for name in names], ylabel="PR-AUC / late rate", ylim=(0, 2), title="PR-AUC over floor: validation vs. test")
    # legend below the axes so it does not cover the value labels
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    style_axes(ax)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="EVAL-4: score the frozen models on the test split, once")
    parser.add_argument("--split", choices=["val", "test"], required=True,
                        help="val = dry run that checks against the saved metrics and writes nothing; test = the one real run")
    args = parser.parse_args()

    # score once: refuse before reading test if it has already been scored
    if args.split == "test" and Path(OUTPUT_PATH).exists():
        raise SystemExit(f"{OUTPUT_PATH} already exists, so test has been scored. Not scoring again.")

    # check every hash before scoring anything, so a bad file can not stop the run halfway through test
    for name, row in FINAL_MODELS.items():
        check_frozen(name, row)

    data, y = load_split(args.split)
    results, probabilities = {}, {}
    for name, row in FINAL_MODELS.items():
        results[name], probabilities[name] = score_model(name, row, data, y)
        if args.split == "val":
            dry_run_check(name, results[name])

    if args.split == "val":
        print("Dry run passed: nothing written, nothing logged.")
        return

    # step 5: save reports/final/eval4_metrics.json, figures, and W&B runs
    results = {name: test_block(result) for name, result in results.items()}
    output = {
        "run": WANDB_RUN,
        "date": str(date.today()),
        "split": "test",
        "rows": int(len(y)),
        "versions": {"pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__},
        "models": results,
    }
    # json first: once it exists the guard blocks a second test run, even if plotting or W&B fails after this line
    save_json(OUTPUT_PATH, output)

    Path(FIGURES_PATH).mkdir(parents=True, exist_ok=True)
    figures = {"test_pr_curves": f"{FIGURES_PATH}/eval4_test_pr_curves.png", "val_vs_test": f"{FIGURES_PATH}/eval4_val_vs_test.png"}
    plot_test_pr_curves(results, probabilities, y, figures["test_pr_curves"])
    plot_val_vs_test(results, figures["val_vs_test"])

    # one W&B run per model, all named EVAL-4; validation + test go to the summary, everything else to config
    for name, block in results.items():
        log_to_wandb({"run": WANDB_RUN, **block}, figures=figures, tags=[FINAL_MODELS[name]["model_tag"], WANDB_FEATURE_TAG, "test", block["role"]])

    for name, block in results.items():
        v, t = block["validation"], block["test"]
        print(f"{name} ({block['role']}): PR-AUC {v['pr_auc']:.4f} -> {t['pr_auc']:.4f}, "
            f"over floor {v['pr_auc_over_floor']:.3f} -> {t['pr_auc_over_floor']:.3f}, F1 {v['f1']:.4f} -> {t['f1']:.4f}")

if __name__ == "__main__":
    main()
