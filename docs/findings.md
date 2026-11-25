# Findings

The story behind the results, written for the report and presentation. Each
section answers one row in [OPEN-ISSUES.md](OPEN-ISSUES.md); that row is the
short record, this is the explanation. Every number here comes from a committed
file or the W&B project `flight-classifier` (team
`mattstaff0-texas-christian-university`); the source is named under each table.

All models use the same 13 base features, train on Jul 2025 – Mar 2026, and are
scored on validation (Apr 2026). The test split (May–Jun 2026) has not been
scored yet (EVAL-4).

## Headline so far

| Model | Validation PR-AUC | × no-skill | F1 | Train vs. validation gap |
|---|---|---|---|---|
| No-skill (predict the late rate for every flight) | 0.202 | 1.00 | 0.336 | — |
| MOD-1 logistic regression | 0.315 | 1.56 | 0.393 | 0.3% |
| MOD-2 XGBoost, default settings | 0.320 | 1.59 | 0.402 | 12.4% |
| **MOD-2 XGBoost, tuned** | **0.332** | **1.64** | **0.407** | 8.3% |

Sources: `reports/logistic_regression/mod1_metrics.json`,
`git show 5363871:reports/xgboost/mod2_metrics.json` (default run),
`reports/xgboost/mod2_metrics.json` (tuned). "× no-skill" is
`pr_auc_over_floor`; the gap is train ÷ validation `pr_auc_over_floor` − 1.

---

## How we measure overfitting (EVAL-3)

**Problem:** raw log loss cannot be compared across our splits. Train is 22.4%
late and April is 20.2% late; with fewer late flights there is less uncertainty,
so even a perfect, non-overfitting model scores a lower log loss on April. For
MOD-1, validation log loss (0.479) was *lower* than train (0.504), which naively
reads as "better on unseen data".

**Fix:** compare each split against its own no-skill model, a model that ignores
every feature and predicts that split's late rate for every flight.
`pr_auc_over_floor` does this for PR-AUC (PR-AUC ÷ late rate), so it is the
number we compare between train and validation.

| | no-skill log loss | model log loss | better than no-skill |
|---|---|---|---|
| MOD-1 train | 0.532 | 0.504 | 5.3% |
| MOD-1 validation | 0.503 | 0.479 | 4.7% |
| MOD-2 default train (round 17) | 0.532 | 0.490 | 7.9% |
| MOD-2 default validation (round 17) | 0.503 | 0.476 | 5.3% |

(Log loss rows computed by hand for illustration; the measured comparison is
`pr_auc_over_floor`.)

**MOD-1 is not overfitting:** `pr_auc_over_floor` 1.567 train vs. 1.562
validation. It is underfit: limited by its features and linear form.

---

## MOD-2: XGBoost

### Setup

- 13 base features. All 9 categorical and time-block columns are pandas
  `category` columns; validation is re-coded to train's category list, so a
  value train never saw becomes missing. City market IDs are converted from
  integers to categories, since they are labels, not quantities.
- **Change from `architecture-design.md`:** the plan was ordinal codes 0–18 for
  `DepTimeBlk`/`ArrTimeBlk`. We kept them as categories for simplicity; a
  category split can still group adjacent blocks. EXP-9 tests whether this
  mattered.
- `XGBClassifier`, `tree_method="hist"`, `enable_categorical=True`. Native
  categorical splits send any *group* of airports left and the rest right, instead
  of one-hot columns (about 1,500 columns, ~60 GB dense) or arbitrary integer
  order.
- Early stopping on validation log loss after 50 rounds without improvement;
  the model keeps the trees up to the best round.
- Fit on train only; test not scored.
- **Unseen airports (PRE-6: 51/48 test rows, CSG and LNS).** After re-coding they
  are missing, and XGBoost sends a missing value down each split's learned
  default branch. Checked on 1,000 validation rows with the four airport/market
  columns set to missing: no errors, mean probability 0.1805 vs. 0.1776 with the
  real airports, range narrowed to 0.134–0.226, correlation 0.80 with the real
  predictions. The flight is predicted like an "average airport" from its other
  features. Re-coding is required, not optional: XGBoost 3.4 stores train's
  category list and rejects an unseen label (`Found a category not in the
  training set`), so test must go through the same `encode()`.

### Step 1: default settings overfit from the first trees

With XGBoost's defaults (`learning_rate` 0.3, `max_depth` 6, no subsampling),
validation log loss was lowest at **round 17** and then rose, while train log
loss kept falling: the textbook overfitting curve.

```
train_logloss █▇▆▅▅▄▄▄▃▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁   falls every round
  val_logloss █▆▅▃▃▂▂▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄   bottoms out at round 17, then rises
```

Validation PR-AUC 0.320 was only slightly above MOD-1's 0.315. The
`pr_auc_over_floor` gap was **12.4%** (1.782 train vs. 1.585 validation), and
that is for the 18-tree model early stopping kept, so the gap was **not** caused
by too many rounds: each tree was learning train-specific patterns from the
start. Early stopping removes the extra trees but not what the first ones
learned, so the fix had to come from the tree settings, not the tree count.

### Step 2: hyperparameter search

Hyperopt (TPE) ran **90 trials**, maximizing validation PR-AUC over 7
hyperparameters. Every trial is logged to W&B (run `MOD-2`, tag
`hyperopt-trial`) and exported to `reports/xgboost/mod2_trials.csv`.

| Hyperparameter | Range searched | Best trial (#74) | Default |
|---|---|---|---|
| `learning_rate` | 0.02 – 0.3 (log) | **0.063** | 0.3 |
| `max_depth` | 3 – 10 | **5** | 6 |
| `min_child_weight` | 1 – 200 (log) | 26 | 1 |
| `subsample` | 0.5 – 1.0 | **0.74** | 1.0 |
| `colsample_bytree` | 0.5 – 1.0 | **0.55** | 1.0 |
| `reg_lambda` | 0.1 – 20 (log) | **7.5** | 1.0 |
| `max_cat_threshold` | 8 – 128 | **120** | 64 |

Early stopping chose 344 trees (vs. 17 with defaults).

**What separated good trials from bad ones** (median of the 10 best vs. the 10
worst of 90, by validation PR-AUC):

| | top 10 | bottom 10 |
|---|---|---|
| `learning_rate` | 0.064 | 0.112 |
| `max_depth` | 5 | 7.5 |
| `reg_lambda` | 6.2 | 1.3 |
| `colsample_bytree` | 0.72 | 0.86 |
| `max_cat_threshold` | 108 | 84 |
| `min_child_weight` | 10.5 | 10.8 |
| train vs. validation gap | 8.6% | 16.5% |

- The winners take **smaller steps, grow shallower trees, carry more L2
  penalty, and see fewer columns per tree**: every one of these makes each tree
  learn less train-specific detail, which matches the step 1 diagnosis. Across
  all 90 trials, the strongest single correlate of validation PR-AUC was the
  number of trees early stopping kept (Spearman 0.56): many small steps beat a
  few big ones.
- **Not everything matched the guess.** `min_child_weight` (minimum data per
  leaf) did not separate good from bad trials (≈10 in both groups; trial #81,
  nearly tied for best, used 2.5). And `max_cat_threshold` went *up*: once each
  tree is kept small by the other settings, letting one split group up to ~120
  categories (e.g. airports) helps.
- **The search reached a plateau.** Every trial beat MOD-1 and 89 of 90 beat the
  default run (range 0.320–0.332, median 0.327). The best score was 0.330 after
  20 trials; the remaining 70 trials added only 0.0015, and the top 5 trials are
  within 0.001 of each other. More trials would not have changed the answer.
- No trial was cut off by the tree cap (most trees needed: 1,363 of 6,000).

### Step 3: the tuned model

| | MOD-1 | MOD-2 default | MOD-2 tuned |
|---|---|---|---|
| Validation PR-AUC | 0.315 | 0.320 | **0.332** |
| `pr_auc_over_floor` (validation) | 1.562 | 1.585 | **1.642** |
| F1 | 0.393 | 0.402 | **0.407** |
| Threshold | 0.233 | 0.222 | 0.229 |
| Precision / recall | 0.288 / 0.621 | 0.290 / 0.656 | 0.297 / 0.647 |
| `pr_auc_over_floor` train vs. validation | 1.567 / 1.562 | 1.782 / 1.585 | 1.778 / 1.642 |
| Gap | 0.3% | 12.4% | 8.3% |
| Trees kept | — | 18 | 345 |

- **Tuning gave 2.4× the improvement the default XGBoost had over the
  baseline** (+0.012 vs. +0.005 PR-AUC).
- **The gap shrank because validation improved, not because train got worse:**
  train `pr_auc_over_floor` stayed at ~1.78 while validation rose from 1.585 to
  1.642. The tuned model learned more of what carries over to April.
- **The validation curve no longer turns up.** It falls and then flattens; early
  stopping now stops at a plateau instead of at the point where the model starts
  getting worse.
- **Some overfitting remains (8.3%).** Trees can learn patterns specific to the
  training months that a linear model cannot, so some gap is expected with an
  out-of-time split. One candidate cause, `DayofMonth` combined with airport
  categories memorizing specific bad days, is tested in EXP-8.

### Caveats

- Hyperparameters, early stopping, and the threshold were all chosen on
  validation, so the validation score is optimistic. The test split (EVAL-4) is
  the unbiased comparison.
- The gain over the baseline is real but modest (PR-AUC 0.315 → 0.332): with
  only booking-time features, much of what makes a flight late (weather, knock-on
  delays) is not in the inputs. Adding historical rates (EXP-4, EXP-1) targets
  that directly.

### Figures

All in `reports/xgboost/figures/`:

- **`mod2_val_logloss_default_vs_tuned.png`** (main slide): validation log loss
  per boosting round, default vs. tuned. Default dips at round 17 and climbs;
  tuned keeps falling and flattens through round 344.
- **`mod2_parallel_coordinates.png`**: all 90 `hyperopt-trial` runs, one line per
  trial across the 7 hyperparameters, colored by validation PR-AUC. Suggested
  slide title: "Best XGBoost trials used small learning rates, shallow trees, and
  more regularization".
- `mod2_default_train_logloss.png`, `mod2_default_val_logloss.png`: the default
  run's curves separately. Train falls every round while validation turns up
  after round 17. If shown together, note that validation sits *below* train only
  because April has fewer late flights: read the shape, not the level.
- `mod2_tuned_train_logloss.png`, `mod2_tuned_val_logloss.png`: the same for the
  tuned run (appendix).
- `mod2_pr_curve.png`, `mod2_thresholds.png`: PR curve and threshold choice for
  the tuned model.

---

## Experiments on MOD-2

Filled in as each one is resolved.

- **EXP-7** (early stopping on PR-AUC instead of log loss): ⟨not run yet⟩
- **EXP-8** (drop `DayofMonth`): ⟨not run yet⟩
- **EXP-9** (ordinal vs. category time blocks): ⟨not run yet⟩
