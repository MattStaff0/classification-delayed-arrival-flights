import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.evaluation import compute_loss, style_axes, BLUE, ORANGE, GRAY
from src.final_evaluation import FINAL_MODELS, check_frozen, load_split, prepare

# EVAL-5: where MOD-2 makes its errors on validation and which features it relies on.
# validation only: load_split is only ever called with "val" here, test is scored only in EVAL-4

OUTPUT_FOLDER = "reports/eval5"
FIGURES_PATH = "reports/eval5/figures"
MIN_FLIGHTS = 1000
GROUP_COLUMNS = ["Reporting_Airline", "Origin", "DepTimeBlk"]

# columns shown for each error case (what the model saw, in readable form)
SHOW_COLUMNS = ["FlightDate", "Reporting_Airline", "Origin", "Dest", "DepTimeBlk", "ArrTimeBlk", "DayOfWeek", "Distance"]
N_CASES = 3          # per kind of error
RANDOM_STATE = 42    # so the random cases are the same every run


def load_model_and_validation():
    # the same frozen MOD-2 file and encoding path EVAL-4 used, so the hash check also proves it is the final model
    row = FINAL_MODELS["MOD-2"]
    check_frozen("MOD-2", row)
    saved = joblib.load(row["model"])
    model = saved["pipeline"]

    data, y = load_split("val")
    prepared, _ = prepare(model, data, row["prep"])
    probabilities = model.predict_proba(prepared)[:, 1]
    return model, saved["threshold"], data, prepared, y, probabilities


def shap_values(model, prepared):
    # one contribution per feature per flight, in log-odds; together with the bias they add up to the prediction
    # iteration_range: the booster also holds the 50 trees grown after the best round (early stopping patience);
    # predict_proba skips them automatically, the raw booster does not, so the range must be given

    contribs = model.get_booster().predict(xgb.DMatrix(prepared, enable_categorical=True), pred_contribs=True, iteration_range=(0, model.best_iteration + 1))

    # the last column is the bias (log-odds of the training late rate), the same for every flight
    return pd.DataFrame(contribs[:, :-1], columns=prepared.columns, index=prepared.index), contribs[:, -1]


def importance_table(model, shap):
    # total_gain: how much each feature improved the training loss, summed over all its splits (while fitting)
    # mean |SHAP|: how far each feature moves the predictions on validation flights different units, so compare shares, not the raw values
      gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
      table = pd.DataFrame({"total_gain": gain, "mean_abs_shap": shap.abs().mean()})
      table["total_gain_share"] = table["total_gain"] / table["total_gain"].sum()
      table["shap_share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
      return table.sort_values("shap_share", ascending=False)


def plot_importance(table, path):
    # reversed so the biggest feature is drawn at the top
    table = table.iloc[::-1]
    y = np.arange(len(table))

    height = 0.4
    figure, ax = plt.subplots(figsize=(7, 5.5))

    ax.barh(y + height / 2, table["total_gain_share"], height, color=GRAY, label="Total gain (train, while fitting)")
    ax.barh(y - height / 2, table["shap_share"], height, color=BLUE, label="Mean |SHAP| (validation, Apr 2026)")
    ax.set(yticks=y, yticklabels=table.index, xlabel="Share of total importance", title="MOD-2 feature importance: train gain vs. validation SHAP")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower right")

    style_axes(ax)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)

def group_table(data, y, probabilities, column, min_flights=MIN_FLIGHTS):
    # one row per value of column (an airline, an airport, a time block), scored on its own flights
    frame = pd.DataFrame({"group": data[column], "late": y, "probability": probabilities})
    rows = []
    for value, group in frame.groupby("group", observed=True):   # observed: skip categories with no April flights
        row = {"group": value,
                "flights": len(group),
                "late_flights": int(group["late"].sum()),
                # mean probability vs. late rate: does the model expect about the right number of late flights here?
                "mean_probability": float(group["probability"].mean())}
        
        # PR-AUC and log loss need both late and on-time flights in the group
        if 0 < row["late_flights"] < row["flights"]:
            row.update(compute_loss(group["late"], group["probability"]))

        row["enough_flights"] = row["flights"] >= min_flights
        rows.append(row)

    # compute_loss calls it floor; here it is the group's late rate
    table = pd.DataFrame(rows).rename(columns={"floor": "late_rate"})

    # groups above the cutoff first, best to worst; small groups after, still in the file
    return table.sort_values(["enough_flights", "pr_auc_over_floor"], ascending=False).reset_index(drop=True)

def error_cases(data, y, probabilities, threshold, n=N_CASES, seed=RANDOM_STATE):
    # false positive: flagged late (probability >= the frozen threshold) but on time; false negative: not flagged but late
    late = y.to_numpy() == 1
    flagged = probabilities >= threshold
    false_positive = flagged & ~late
    false_negative = ~flagged & late

    # row positions sorted from lowest to highest probability
    order = np.argsort(probabilities)
    rng = np.random.default_rng(seed)
    picks = {
        # confident mistakes show blind spots: highest-probability on-time flights, lowest-probability late flights
        "confident_false_positive": order[false_positive[order]][::-1][:n],
        "confident_false_negative": order[false_negative[order]][:n],
        # random mistakes show what a typical error looks like
        "random_false_positive": rng.choice(np.flatnonzero(false_positive), n, replace=False),
        "random_false_negative": rng.choice(np.flatnonzero(false_negative), n, replace=False),
    }

    cases = []
    for kind, rows in picks.items():
        for row in rows:
            cases.append({"kind": kind, "row": int(row), "probability": float(probabilities[row]),
                        "late": int(late[row]), **data.iloc[row][SHOW_COLUMNS].to_dict()})
    return pd.DataFrame(cases)


def top_reasons(shap, rows, top=3):
    # the features that pushed each prediction the most, in log-odds (+ toward late, - toward on time)
    reasons = []
    for row in rows:
        contributions = shap.iloc[row].sort_values(key=abs, ascending=False).head(top)
        reasons.append(", ".join(f"{feature} {value:+.2f}" for feature, value in contributions.items()))
    return reasons

def main():
    parser = argparse.ArgumentParser(description="EVAL-5: where MOD-2 makes its errors on validation (never test)")
    parser.add_argument("--min-flights", type=int, default=MIN_FLIGHTS, help="smallest group that is ranked and quoted; smaller groups stay in the CSV")
    args = parser.parse_args()

    model, threshold, data, prepared, y, probabilities = load_model_and_validation()
    Path(FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    # feature importance: SHAP for every validation flight (about 40 s), then train gain next to it
    shap, _ = shap_values(model, prepared)
    importance = importance_table(model, shap)
    importance.to_csv(f"{OUTPUT_FOLDER}/eval5_feature_importance.csv", index_label="feature")
    plot_importance(importance, f"{FIGURES_PATH}/eval5_feature_importance.png")
    print("Feature importance (share):")
    print(importance[["total_gain_share", "shap_share"]].round(3).to_string())

    # one table per grouping column; print the best and worst three among groups above the cutoff
    for column in GROUP_COLUMNS:
        table = group_table(data, y, probabilities, column, args.min_flights)
        table.to_csv(f"{OUTPUT_FOLDER}/eval5_groups_{column}.csv", index=False)
        ranked = table[table["enough_flights"]]
        share = ranked["flights"].sum() / table["flights"].sum()
        print(f"\n{column}: {len(ranked)} of {len(table)} groups have >= {args.min_flights:,} flights ({share:.1%} of validation flights)")
        shown = ["group", "flights", "late_rate", "mean_probability", "pr_auc", "pr_auc_over_floor"]
        print(pd.concat([ranked.head(3), ranked.tail(3)])[shown].round(3).to_string(index=False))

     # representative errors at the frozen threshold, with the features that pushed each prediction most
    cases = error_cases(data, y, probabilities, threshold)
    cases["top_reasons"] = top_reasons(shap, cases["row"])
    cases.to_csv(f"{OUTPUT_FOLDER}/eval5_error_cases.csv", index=False)
    print(f"\nError cases at threshold {threshold:.4f}:")

    # round only probability: FlightDate is a date and .round() on it raises a warning
    print(cases.drop(columns=["row", "late"]).round({"probability": 3}).to_string(index=False))


if __name__ == "__main__":
    main()