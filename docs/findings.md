# Findings

> **AI-assistance notice:** This document was written with help from an AI
> assistant (Claude, via Claude Code). The research questions, the choices
> between approaches, and the conclusions are the team's. The assistant reformatted
> and edited the text, proposed some methods and checks (for example, the
> test-set freeze and the SHAP comparison), ran read-only checks, and formatted
> the tables. Every number comes from a committed results file or W&B run named
> next to it. The team reviewed the content and is responsible for it. It is
> acknowledged here per the course's AI-use policy.

The story behind the results, written for the report and presentation. Each
section answers one row in [OPEN-ISSUES.md](OPEN-ISSUES.md); that row is the
short record, this is the explanation. Every number here comes from a committed
file or the W&B project `flight-classifier` (team
`mattstaff0-texas-christian-university`); the source is named under each table.

All models use the same 13 base features, train on Jul 2025 – Mar 2026 (except
EXP-2, which tests a longer training window), and are scored on validation
(Apr 2026). The test split (May–Jun 2026) was scored once, at the end, for
the frozen final models (EVAL-4, below the headline table and in its own section).

## Headline so far

| Model | Validation PR-AUC | × no-skill | F1 | Train vs. validation gap |
|---|---|---|---|---|
| No-skill (predict the late rate for every flight) | 0.202 | 1.00 | 0.336 | — |
| MOD-1 logistic regression | 0.315 | 1.56 | 0.393 | 0.3% |
| MOD-2 XGBoost, default settings | 0.320 | 1.59 | 0.402 | 12.4% |
| **MOD-2 XGBoost, tuned** | **0.332** | **1.64** | **0.407** | 8.3% |
| MOD-3 LightGBM, default settings | 0.319 | 1.58 | 0.402 | 13.6% |
| MOD-3 LightGBM, tuned (7 leaves) | 0.331 | 1.64 | 0.408 | 8.4% |
| EXP-4 logistic regression + late rate | 0.318 | 1.57 | 0.396 | 0.8% |
| EXP-4 XGBoost, tuned + late rate | 0.332 | 1.64 | 0.407 | 7.5% |
| EXP-2 XGBoost, tuned, train Jul 2024 – Mar 2026 (21 months) | 0.326 | 1.61 | 0.404 | 6.1% |
| EXP-2 run B: XGBoost, tuned, train Apr 2025 – Mar 2026 (12 months) | 0.331 | 1.64 | 0.406 | 7.7% |

Sources: `reports/logistic_regression/mod1_metrics.json`,
`reports/logistic_regression/exp4_metrics.json`, `reports/xgboost/exp4_metrics.json` (EXP-4),
`reports/xgboost/exp2_metrics.json`, `reports/xgboost/exp2_aprjun_metrics.json` (EXP-2),
`git show 5363871:reports/xgboost/mod2_metrics.json` (default run),
`reports/xgboost/mod2_metrics.json` (tuned),
`git show 16fd05d:reports/lightgbm/mod3_metrics.json` (MOD-3 default run),
`reports/lightgbm/mod3_metrics.json` (MOD-3 tuned; the search's own best,
trial 26, is `git show e3351b6:reports/lightgbm/mod3_metrics.json`). "× no-skill" is
`pr_auc_over_floor`; the gap is train ÷ validation `pr_auc_over_floor` − 1.

### Final result: test (EVAL-4)

Scored once on May–Jun 2026 (1,199,540 flights) at each model's frozen
validation threshold. Test has more late flights than validation (0.2355 vs.
0.202), so compare "× no-skill", not raw PR-AUC or F1.

| Model | Validation PR-AUC | Test PR-AUC | Validation × no-skill | Test × no-skill | Validation F1 | Test F1 |
|---|---|---|---|---|---|---|
| No-skill | 0.202 | 0.236 | 1.00 | 1.00 | 0.336 | 0.381 |
| MOD-1 logistic regression (baseline) | 0.315 | 0.352 | 1.56 | 1.49 | 0.393 | 0.436 |
| **MOD-2 XGBoost, tuned (final)** | **0.332** | **0.369** | **1.64** | **1.56** | **0.407** | **0.446** |
| MOD-3 LightGBM, tuned (reference only) | 0.331 | 0.367 | 1.64 | 1.56 | 0.408 | 0.447 |

Source: `reports/final/eval4_metrics.json` (W&B runs `EVAL-4`: `qiwzflds`,
`o4n36jg2`, `ubpfkx8j`).

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

## MOD-3: LightGBM

**Question:** with the same features, encoding, and splits as MOD-2, how does
LightGBM compare with XGBoost in validation PR-AUC, F1, overfitting, and
training time? Only the algorithm changes.

### Setup

Same as MOD-2: 13 base features, the same `encode()` (pandas `category`
columns, validation re-coded to train's category list), train on Jul 2025 – Mar
2026, score on validation, test not scored, early stopping on validation log
loss after 50 rounds, tree cap 6,000. What differs:

- **How trees grow.** XGBoost grows each tree level by level, so `max_depth`
  sets its size. LightGBM grows **leaf by leaf**: it always splits the one leaf
  that reduces the loss most, so trees can be deep and uneven. Size is set by
  **`num_leaves`** instead; `max_depth` stays at −1 (no limit). For scale, MOD-2's
  tuned depth of 5 allows at most 32 leaves, about LightGBM's default of 31.
- **Minimum leaf size is a row count.** LightGBM's `min_child_samples` counts
  rows per leaf. XGBoost's `min_child_weight` is a sum of hessians, p(1 − p) per
  row, about 0.16 at our ~20% late rate, so the two are on different scales.
- **No L2 penalty by default.** LightGBM's default `reg_lambda` is 0 (XGBoost's
  is 1). MOD-2's search found that more L2 helped, so the default run starts with
  none.
- **Row sampling has to be switched on.** `subsample` does nothing unless
  `subsample_freq` > 0, so `subsample_freq=1` (resample rows every tree) is a
  fixed setting. With `subsample=1.0` the default run still uses every row.
- **Early stopping is a callback.** `lgb.early_stopping(50)` is passed to
  `fit()` and watches validation only (the training set in the eval list is
  ignored for stopping; it is there to record train log loss).
  `lgb.log_evaluation(100)` prints scores every 100 rounds. Eval sets are passed
  as `eval_X`/`eval_y` because `eval_set` is deprecated in LightGBM 4.7.
- **Tree counts are stored 0-based, like XGBoost.** LightGBM's
  `best_iteration_` counts from 1 and XGBoost's from 0, so MOD-3's JSON stores
  `best_iteration_ − 1`. For both models: trees kept = `best_iteration` + 1, and
  `rounds_trained` = trees kept + 50 patience rounds. The model itself is not
  changed; `predict_proba` still uses all kept trees.
- **Training time** (`train_seconds`) times `fit()` only: not data loading,
  scoring, or W&B logging. Both libraries use all CPU cores by default and also
  compute train and validation log loss every round.
- **Unseen airports (PRE-6, test only).** A LightGBM categorical split lists the
  categories that go left; **everything else goes right, including missing
  values**. XGBoost instead learns a default direction for missing values at
  each split. LightGBM also stores train's category list and silently turns an
  unseen label into missing, where XGBoost raises an error. Checked on a toy
  model: a missing value and an unseen label `Z` both got the prediction of the
  "everything else" side. MOD-3 still uses `encode()`, so both models get
  identical input.

### Step 1: default settings overfit about as much as XGBoost's

LightGBM's defaults: `learning_rate` 0.1, `num_leaves` 31, `min_child_samples`
20, `reg_lambda` 0, no row or column sampling, `max_cat_threshold` 32,
`cat_smooth` 10.

Validation log loss was lowest at **round 60** (0.47610). Training continued
for the 50-round patience window and stopped after round 110. At round 100,
train log loss was still falling (0.4908 → 0.4858) while validation had turned
slightly up (0.47610 → 0.47634): the same overfitting shape as XGBoost's
default, later because the learning rate is smaller (0.1 vs. 0.3).

```
train_logloss █▇▇▇▆▅▄▄▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁   falls every round
  val_logloss █▇▆▄▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁   flattens; the rise after round 60 is too small to show
```

**Validation log loss below train does not mean "no overfitting".** It is 0.476
vs. 0.491 only because April has fewer late flights (20.2% vs. 22.4%), the same
effect as in EVAL-3. The `pr_auc_over_floor` gap, which adjusts for each split's
late rate, is **13.6%**, slightly above XGBoost's default 12.4%.

| | MOD-1 | MOD-2 default | MOD-3 default |
|---|---|---|---|
| Validation PR-AUC | 0.315 | 0.320 | 0.319 |
| `pr_auc_over_floor` (validation) | 1.562 | 1.585 | 1.578 |
| F1 | 0.393 | 0.402 | 0.402 |
| Threshold | 0.233 | 0.222 | 0.224 |
| Precision / recall | 0.288 / 0.621 | 0.290 / 0.656 | 0.291 / 0.647 |
| `pr_auc_over_floor` train vs. validation | 1.567 / 1.562 | 1.782 / 1.585 | 1.793 / 1.578 |
| Gap | 0.3% | 12.4% | 13.6% |
| Trees kept | — | 18 | 60 |
| Training time | — | not recorded | 8.5 s (110 rounds, 0.077 s/round) |

Source: `git show 16fd05d:reports/lightgbm/mod3_metrics.json`, W&B run
`xr932vla`.

- **With defaults, the two libraries are tied** on PR-AUC (0.319 vs. 0.320) and
  F1 (0.402 for both).
- **As with MOD-2, the gap is not caused by too many trees.** Early stopping
  already kept only the best 60. The fix has to come from the tree settings:
  smaller trees, larger minimum leaf size, more L2. That is what the search
  covers.

### Step 2: hyperparameter search

Hyperopt TPE, 90 trials (the same budget as MOD-2), maximizing validation
PR-AUC. Each trial is logged to W&B as `MOD-3` with tag `hyperopt-trial`, and
all trials are exported to `reports/lightgbm/mod3_trials.csv`. Same 7 settings
as MOD-2, using LightGBM's equivalents and the same ranges where they exist:

| Hyperparameter | Range searched | Best trial (#26) | Default |
|---|---|---|---|
| `learning_rate` | 0.02 – 0.3 (log) | **0.037** | 0.1 |
| `num_leaves` (replaces `max_depth`) | 15 – 255 (log, integer) | **15** (lower bound) | 31 |
| `min_child_samples` (replaces `min_child_weight`) | 20 – 2,000 (log, integer) | **692** | 20 |
| `subsample` | 0.5 – 1.0 | **0.77** | 1.0 |
| `colsample_bytree` | 0.5 – 1.0 | **0.55** | 1.0 |
| `reg_lambda` | 0.1 – 20 (log) | 0.79 | 0 |
| `max_cat_threshold` | 8 – 128 (step 8) | 88 | 32 |

`max_depth` (−1) and `cat_smooth` (10) stay at their defaults. Trial 26 scored
validation PR-AUC 0.329 (`pr_auc_over_floor` 1.630, F1 0.406, 11.1% gap, 948
trees). Source: `reports/lightgbm/mod3_trials.csv`; trial 26's final run is
`git show e3351b6:reports/lightgbm/mod3_metrics.json` (W&B run `8cpfqsss`,
tagged `superseded`).

**Why a log scale for `num_leaves` and `min_child_samples`:** their effect is
relative. Going from 15 to 30 leaves changes the model far more than going from
225 to 240. TPE starts with 20 random trials; on a uniform scale only about 6%
of `num_leaves` draws fall in 15–30 (the size of MOD-2's best trees), and on a
log scale about 25% do (checked on 2,000 draws).

**What separated good trials from bad ones** (median of the 10 best vs. the 10
worst by validation PR-AUC):

| | top 10 | bottom 10 |
|---|---|---|
| `num_leaves` | **16.5** | 99.5 |
| `learning_rate` | 0.023 | 0.063 |
| `min_child_samples` | 739 | 111 |
| `colsample_bytree` | 0.54 | 0.91 |
| `subsample` | 0.56 | 0.77 |
| `reg_lambda` | 0.82 | 2.25 |
| `max_cat_threshold` | 92 | 68 |
| train vs. validation gap | 10.5% | 19.5% |

- **Tree size mattered most.** Every top-10 trial had 15–19 leaves. Rank
  correlation (Spearman) of `num_leaves` with validation PR-AUC is −0.63, and
  with the gap +0.75: smaller trees scored higher *and* overfit less. Next
  strongest: `colsample_bytree` (−0.58), `subsample` (−0.38), and
  `learning_rate` (−0.37).
- **The good trials sit in the most constrained corner of the space:** small
  trees, slow learning, large minimum leaves, and few columns and rows per tree.
  This is the same direction as MOD-2 (shallower, slower, fewer columns per
  tree), only stronger.
- **Unlike MOD-2, L2 did not matter much** (`reg_lambda` correlation +0.11; top
  10 median 0.82 vs. MOD-2's best 7.5). Here the regularization comes from
  small trees and large leaves instead.
- **Results by tree size:**

  | `num_leaves` | trials | best PR-AUC | median PR-AUC | median gap |
  |---|---|---|---|---|
  | 15–20 | 25 | **0.3292** | **0.3272** | **11.1%** |
  | 21–32 | 25 | 0.3276 | 0.3242 | 13.8% |
  | 33–64 | 19 | 0.3264 | 0.3222 | 16.5% |
  | 65–128 | 12 | 0.3255 | 0.3211 | 17.6% |
  | 129–255 | 9 | 0.3244 | 0.3209 | 18.4% |

- All 90 trials beat MOD-1 (0.315), 83 beat the MOD-3 default (0.3188), and
  **none beat MOD-2 tuned (0.3317)**; range 0.3164–0.3292.
- **The search plateaued early:** trial 21 was already within 0.001 of the best
  (trial 26).
- No trial was cut off by the tree cap (most trees used: 1,334 of 6,000).

### Step 2b: the best trial was at the edge of the range, so we checked below it

Trial 26 used `num_leaves` = 15, the **smallest value the search allowed**, and
the score was still improving toward that edge (table above). When the best
value sits on a boundary, the true optimum may be past it. We re-ran trial 26's
settings with **only `num_leaves` changed** (4, 7, 10, and 15 as the
reference), logged to W&B as `MOD-3` with tag `boundary-check`:

| `num_leaves` | Validation PR-AUC | `pr_auc_over_floor` val | `pr_auc_over_floor` train | Gap | F1 | Trees kept |
|---|---|---|---|---|---|---|
| 15 (trial 26) | 0.3292 | 1.630 | 1.811 | 11.1% | 0.406 | 948 |
| 10 | 0.3302 | 1.635 | 1.788 | 9.4% | 0.407 | 1,545 |
| **7** | **0.3308** | **1.638** | 1.775 | 8.4% | **0.408** | 2,494 |
| 4 | 0.3287 | 1.627 | 1.711 | 5.1% | 0.406 | 3,049 |

- **The optimum is inside the range: 7 leaves.** The score rises from 15 to 7
  and falls at 4.
- **At 4 leaves the model underfits.** It has the smallest gap (5.1%), but only
  because it is too simple to fit even the training months (train
  `pr_auc_over_floor` drops to 1.711), and validation drops too. A smaller gap
  only helps when validation does not get worse.
- The 15-leaf run reproduced trial 26 exactly (best round 947, same validation
  log loss), so the comparison is like for like.
- **Final MOD-3 configuration: trial 26 with `num_leaves` = 7**, trained with
  `--params reports/lightgbm/mod3_metrics.json --set num_leaves=7`. The settings
  and override are recorded in `mod3_metrics.json` (`params_source`). The search
  space in the code stays 15–255, the range that was actually searched.

### Step 3: the tuned model vs. XGBoost

| | MOD-2 tuned | MOD-3 default | MOD-3 tuned (7 leaves) |
|---|---|---|---|
| Validation PR-AUC | **0.332** | 0.319 | 0.331 |
| `pr_auc_over_floor` (validation) | **1.642** | 1.578 | 1.638 |
| F1 | 0.407 | 0.402 | **0.408** |
| Threshold | 0.229 | 0.224 | 0.228 |
| Precision / recall | 0.297 / 0.647 | 0.291 / 0.647 | 0.297 / 0.652 |
| `pr_auc_over_floor` train vs. validation | 1.778 / 1.642 | 1.793 / 1.578 | 1.775 / 1.638 |
| Gap | **8.3%** | 13.6% | 8.4% |
| Trees kept | 345 | 60 | 2,494 |
| Rounds trained | 395 | 110 | 2,544 |
| Training time | **40.7 s** (median of 3) | 8.5 s (1 run) | 145.4 s (median of 3) |
| Seconds per round | 0.103 | 0.077 | **0.057** |

Source: `reports/lightgbm/mod3_metrics.json` (W&B run `3k15e6d7`),
`reports/xgboost/mod2_metrics.json`.

- **Tuning helped both libraries by the same amount:** +0.012 PR-AUC for each
  (LightGBM 0.319 → 0.331, XGBoost 0.320 → 0.332).
- **On validation the two are tied.** XGBoost is 0.001 ahead on PR-AUC,
  LightGBM is 0.001 ahead on F1, and the gaps are 8.3% vs. 8.4%. These
  differences are far smaller than either model's gain from tuning, and both
  were tuned on validation, so validation cannot separate them.
- **The gap shrank mostly because validation improved:** validation
  `pr_auc_over_floor` rose from 1.578 to 1.638 while train moved only from 1.793
  to 1.775. The same pattern as MOD-2.
- **LightGBM reached the same score with very different trees:** 2,494 trees of
  7 leaves (roughly depth-3 trees) against XGBoost's 345 trees of depth up to 5
  (up to 32 leaves). Many small, slow steps instead of fewer larger ones.
- **The validation curve no longer turns up.** It falls and flattens through
  round 2,494 (`mod3_val_logloss_default_vs_tuned.png`), where the default run
  bottomed at round 60.
- **XGBoost trains the final model about 3.6× faster** (40.7 s vs. 145.4 s),
  even though each LightGBM round is about 1.8× cheaper (0.057 s vs. 0.103 s).
  The tuned LightGBM needs 6.4× as many rounds (2,544 vs. 395) because its trees
  are tiny and its learning rate is low. Leaf-wise growth is faster per tree, as
  MOD-3's question expected, but the configuration that scores best gives that
  advantage back in tree count.
- **Verdict: XGBoost (MOD-2) is the tree model for EXP-2, EXP-4, and EVAL-5.**
  The scores are tied on validation, XGBoost is 0.001 ahead on PR-AUC, and it
  trains the final configuration 3.6× faster, which matters most for EXP-2's
  roughly doubled training set.

### Training time

Each final configuration was trained 3 times with `--params` (MOD-2:
`reports/xgboost/mod2_metrics.json`; MOD-3: `reports/lightgbm/mod3_metrics.json`,
7 leaves), alternating the models so any slowdown over the session hits both
equally. W&B was off (`WANDB_MODE=disabled`) so no extra runs were logged; the
laptop was plugged in with other apps closed. `train_seconds` times `fit()` only.
Every run reproduced its model exactly (XGBoost: best iteration 344, PR-AUC
0.332; LightGBM: best iteration 2,493, PR-AUC 0.331).

| Run (in order) | XGBoost (395 rounds) | LightGBM (2,544 rounds) | LightGBM ÷ XGBoost |
|---|---|---|---|
| 1 | 31.5 s | 121.5 s | 3.9× |
| 2 | 40.7 s | 145.4 s | 3.6× |
| 3 | 46.7 s | 163.9 s | 3.5× |
| **Median** | **40.7 s** | **145.4 s** | **3.6×** |

- **Both models slowed down run after run** (XGBoost +48%, LightGBM +35% from
  run 1 to run 3). The likely cause is the laptop throttling as it heats up (a
  fanless MacBook Air), which is why the runs alternated.
- **The ratio stayed stable** (3.5–3.9×) even though the absolute times did not,
  so the comparison holds; the seconds themselves depend on the machine.
- The metrics JSONs keep the `train_seconds` of their W&B-logged final runs
  (MOD-3: 124.9 s; MOD-2's final run predates the timer). The timing runs were
  not committed, so the JSONs still match their W&B runs.

Command (from the project root):

```bash
for i in 1 2 3; do
  WANDB_MODE=disabled uv run python -m src.ensemble.xgboost_model --params reports/xgboost/mod2_metrics.json
  WANDB_MODE=disabled uv run python -m src.ensemble.lightgbm_model --params reports/lightgbm/mod3_metrics.json
done
```

Afterwards `git restore reports/`: each `--params` run rewrites the metrics
JSON (same scores, new `date` and `train_seconds`).

### Caveats

- Hyperparameters, early stopping, and the threshold were all chosen on
  validation, so validation scores are optimistic. The test split (EVAL-4) is
  the unbiased comparison.
- The boundary check is additional tuning on validation, and it changed only
  `num_leaves` with the other six settings fixed at trial 26's values. With
  7-leaf trees the other settings might have a different optimum; a new search
  with a wider range would test that.
- `learning_rate`, `subsample`, and `colsample_bytree` also had their best
  trials near the low end of their ranges (top-10 medians 0.023, 0.56, 0.54).
  Only `num_leaves` was checked past its bound.
- Training times come from one laptop, using all cores, with train and
  validation log loss computed every round for both models. **Single timings are
  noisy:** the identical 7-leaf configuration took 173.9 s in the boundary check
  and 124.9 s in the final run, and 121.5–163.9 s across the 3 timing runs. The
  comparison therefore uses the median of 3 alternating runs per model, and
  reports seconds per round as well as total time, since the two models stop at
  different rounds. The ratio between the models (3.5–3.9×) is more reliable
  than the absolute seconds.
- The speed result is for the **tuned** configurations. With default settings
  LightGBM trained in 8.5 s (one run); the tuned model is slow because the search
  chose small trees and a low learning rate, not because LightGBM is slow.

### Figures

All in `reports/lightgbm/figures/`:

- `mod3_default_train_logloss.png`, `mod3_default_val_logloss.png`: the default
  run's curves (W&B panels). As with MOD-2, validation sits below train only
  because April has fewer late flights: read the shape, not the level.
- **`mod3_val_logloss_default_vs_tuned.png`** (main slide): validation log loss
  per boosting round, default vs. tuned. Default bottoms at round 60 and stops at
  110; tuned keeps falling and flattens through round 2,494.
- **`mod3_parallel_coordinates.png`**: all 90 `hyperopt-trial` runs (boundary
  check and final runs excluded), one line per trial across the 7 searched
  settings, colored by validation PR-AUC. Suggested slide title: "Best LightGBM
  trials used the smallest trees, slow learning, and large leaves".
- `mod3_tuned_train_logloss.png`, `mod3_tuned_val_logloss.png`: the final
  7-leaf run's curves separately (appendix).
- `mod3_pr_curve.png`, `mod3_thresholds.png`: PR curve and threshold choice for
  the final 7-leaf model. The default run's versions are in commit `16fd05d`.
- Boundary check: no figure; use the Step 2b table. Plotting validation PR-AUC
  against `num_leaves` shows the peak at 7; a train-only chart does not show the
  trade-off.

---

## Experiments

Filled in as each one is resolved. In scope for the course: **EXP-4**
(historical late-rate feature, on logistic regression and the best tree model)
and **EXP-2** (full-year training data, on the best tree model with its tuned
settings fixed). EXP-7, EXP-8, and EXP-9 (the single-run XGBoost follow-ups) are
deferred to the Future work section of `OPEN-ISSUES.md`.

- **EXP-4** (historical late-rate feature): small gain for logistic regression,
  none for XGBoost. Not adopted. See below.
- **EXP-2** (full-year training data): no gain. A full extra year lowers
  validation PR-AUC (0.332 → 0.326); adding only Apr–Jun 2025 ties MOD-2. Not
  adopted: MOD-2 on Jul 2025 – Mar 2026 stays the tree model. See below.

### EXP-4: historical late rate (`Origin` + `DepTimeBlk`)

**Question:** does adding one derived feature, the historical share of late
flights for the flight's origin airport and departure time block, to the 13 base
features improve validation PR-AUC and F1, for logistic regression (MOD-1) and
for XGBoost with MOD-2's tuned settings fixed? Only the feature changes.

#### Setup

**What stays fixed.** Same splits and rows (5,105,534 train, 591,206
validation), same encoding of the 13 base columns, same settings (LR: `C` 1.0,
`tol`, `max_iter`; XGBoost: `--params reports/xgboost/mod2_metrics.json`, no
overrides, recorded as `params_source`), same early-stopping rule (validation
log loss, patience 50), and the same threshold rule (max F1 on validation).
Early stopping may keep a different number of trees: the rule is fixed, the tree
count is not. Re-tuning with the new feature would change two things at once, so
the code rejects `--late-rate` without `--params`. Baselines are the committed
MOD-1 and MOD-2 JSONs; re-running both with the flag off reproduced them exactly
(PR-AUC, F1, threshold, log loss, train PR-AUC, features), so the code changes
did not alter them.

**The feature.** For key *k* = (`Origin`, `DepTimeBlk`):

  late_rate(*k*) = (late flights in *k* + α · *p*) / (flights in *k* + α), α = 20

where *p* is the overall late rate of the rows the counts came from. Train has
4,460 keys (356 origins × 19 blocks, of the 6,764 possible). Code:
`src/features/late_rate.py`, shared by both models.

**Why a naive rate leaks.** If each training row's rate is computed from all of
train, the row's own label is part of its own feature. A key with 1 flight gets
exactly its label; a key with 3 flights gets a value that is a third that
label. Train has 125 keys with 1 flight and 284 with ≤ 5 (72% of those have a
rate of exactly 0 or 1). Validation rows look up honest past rates, so the model
learns to trust the feature more on train than it should on validation.

**Why leave-one-out still leaks.** Leaving out only the row itself gives
(key's late count − own label) / (flights − 1). Everything but the own label is
shared by the key, so late flights always get a slightly *lower* value than
on-time flights in the same key: the label is encoded backwards. In all 4,225
train keys with both outcomes, the late flights got the lower value 100% of the
time (by 0.15 on average in keys of 2–30 flights, 0.006 in larger keys). The
model also sees `Origin` and `DepTimeBlk`, so it can read the label from that.

**Out-of-fold (K = 5).** Train is split into 5 random folds (`KFold`,
`shuffle=True`, `random_state=42`). Each fold's rows get rates computed from the
other 4 folds only, including *p*. All rows in the same fold and key get the same
value whatever their own labels. The seed is fixed in the shared module, so both
models get identical values. **Validation (and later test) rows** get rates from
all of train: their labels never enter the table, so there is nothing to leak.

**More folds is not better.** Smaller folds bring back the leave-one-out
reversal. Within-key correlation between label and feature (0 = no leak):

| Folds | 5 | 10 | 50 | 500 | leave-one-out |
|---|---|---|---|---|---|
| corr(label, rate) within key | −0.055 | −0.081 | −0.172 | −0.311 | −0.358 |

K = 5 cuts it to about a sixth of leave-one-out, not to zero (see Caveats).

**Small and unseen keys: smoothing.** α adds 20 pretend flights at rate *p* to
every key: a key with few flights is pulled toward *p*, a large key keeps its own
rate (2 flights, both late: 1.000 → 0.295; 5,000 flights, 1,500 late: 0.300 →
0.2997). A key with no flights gets exactly *p*, so keys unseen in train (38
validation keys, 383 rows) and keys that land entirely in one fold need no
special case and never give 0/0. α was fixed before any run, from train only:
one flight's variance *p*(1 − *p*) ≈ 0.174 divided by the between-key variance of
true rates (standard deviation ≈ 0.089, after removing sampling noise) gives
α ≈ 22, rounded to 20. Tuning α on validation would add a second choice made on
validation. It barely matters: only 0.075% of train rows are in keys with fewer
than 20 flights (under 1% even at α = 100).

**Smoothing does not replace out-of-fold.** Smoothing the naive rate still
leaks: a 1-flight key gets 0.261 if late and 0.213 if on time. Out-of-fold fixes
leakage; smoothing fixes noise in small keys; EXP-4 uses both.

**Effect on the feature alone.** Used as a score by itself, the naive rate gives
`pr_auc_over_floor` 1.465 train / 1.437 validation; the out-of-fold, smoothed
rate gives 1.451 / 1.442: less inflated on train, slightly better on validation.

**Encoding.** XGBoost: a numeric column (trees split on thresholds, so scale does
not matter). Logistic regression: added to `SCALED_COLUMNS` (`StandardScaler`
fitted on train's out-of-fold values). Unscaled, the rate varies by only about
±0.09, so it would need a large coefficient, which the L2 penalty shrinks; scaling
puts it on the same footing as `Distance`. The rate is computed before
`encode()`, on the raw columns, so every key is looked up by its raw value and a
key with no training flights gets *p* from the formula, whatever the encoding
does with it. (XGBoost's `encode()` does not turn test's unseen airports CSG and
LNS into missing values: `preprocess.py` builds the `Origin` category list from all 12
months, so they keep a code with no training rows. Only the city market IDs,
which `encode()` makes categories from train alone, become missing. Found in
EXP-2; validation has no unseen values either way, so EXP-4's results are
unaffected.)

#### Results

| | MOD-1 | EXP-4 LR | MOD-2 tuned | EXP-4 XGBoost |
|---|---|---|---|---|
| Validation PR-AUC | 0.3154 | **0.3180** (+0.0025) | **0.3317** | 0.3315 (−0.0002) |
| `pr_auc_over_floor` (validation) | 1.562 | 1.574 | 1.642 | 1.641 |
| F1 | 0.393 | **0.396** (+0.003) | 0.4072 | 0.4073 (+0.0001) |
| Threshold | 0.233 | 0.221 | 0.229 | 0.230 |
| Precision / recall | 0.288 / 0.621 | 0.285 / 0.650 | 0.297 / 0.647 | 0.297 / 0.650 |
| `pr_auc_over_floor` train vs. validation | 1.567 / 1.562 | 1.586 / 1.574 | 1.778 / 1.642 | 1.764 / 1.641 |
| Gap | 0.3% | 0.8% | 8.3% | **7.5%** |
| Iterations / trees kept | 50 iterations | 44 iterations | 345 trees | 302 trees |

Source: `reports/logistic_regression/exp4_metrics.json` (W&B run `1r7xbrnu`,
`EXP-4-logistic-regression`), `reports/xgboost/exp4_metrics.json` (W&B run
`4vwq1ui5`, `EXP-4-xgboost`), both tagged `13-plus-late-rate`; baselines as in
the headline table.

- **Logistic regression gains a little.** LR with one-hot `Origin` and one-hot
  `DepTimeBlk` is additive: it learns "DFW adds x" and "17:00 adds y" but cannot
  represent "DFW *at* 17:00 is especially bad". The late rate gives it that
  interaction as one number. +0.0025 PR-AUC is about a fifth of what tuning gave
  XGBoost (+0.012) and larger than the 0.001 we called a tie in MOD-3.
- **XGBoost gains nothing on validation.** Trees can already build the
  `Origin` × `DepTimeBlk` interaction from the two columns. −0.0002 PR-AUC and
  +0.0001 F1 are noise.
- **But XGBoost's gap shrank from 8.3% to 7.5%, from train, not validation.**
  Train `pr_auc_over_floor` fell from 1.778 to 1.764 while validation stayed at
  1.641, and early stopping kept fewer trees (345 → 302). A plausible reading:
  the trees had been spending splits on `Origin` × `DepTimeBlk` combinations,
  some specific to the training months; given the rate as one number, they need
  fewer of them. Same validation score, slightly less overfitting.
- **LR's gap grew from 0.3% to 0.8%.** Still a tenth of XGBoost's and not
  overfitting. Expected from the two known mismatches between train and
  validation values (Caveats).

**Verdict: not adopted.** The late rate helps the linear model a little and adds
nothing to XGBoost, which stays the tree model on 13 features for EXP-2, EVAL-5,
and EVAL-4. EVAL-4 scores MOD-1 as the baseline; the EXP-4 LR result is reported
as the ablation. `src/features/late_rate.py` stays for EXP-1 (weather-delay
rate), which needs the same out-of-fold and smoothing logic.

#### Caveats

- **Out-of-fold leaves a small trace.** Within a key the 5 folds get 5 slightly
  different values, and a fold that happens to hold more late flights gets a
  slightly lower rate (within-key correlation −0.055, table above). Validation
  rows do not have this, so a model that used it would show a larger gap.
- **Train and validation values come from different amounts of data:** training
  rows' rates use 80% of train (4 folds), validation's use 100%. This matters
  mainly for small keys, which smoothing covers.
- One key only (`Origin` + `DepTimeBlk`). The data dictionary also lists
  `Reporting_Airline` + flight number as a key; not tested.
- The differences are small and both runs use validation for early stopping and
  the threshold. A bootstrap of validation PR-AUC would put an interval on LR's
  +0.0025; not done. Test (EVAL-4) is the unbiased comparison.
- Single run per model; `train_seconds` (28.3 s for EXP-4 XGBoost vs. 40.7 s
  median for MOD-2) is not comparable, because single timings on this laptop
  vary widely (MOD-3 Caveats).

#### Figures

- `reports/logistic_regression/figures/exp4_pr_curve.png`, `exp4_thresholds.png`
- `reports/xgboost/figures/exp4_pr_curve.png`, `exp4_thresholds.png`
- W&B: filter by tag `13-plus-late-rate`; `EXP-4-xgboost` has per-round
  `train_logloss` / `val_logloss` history to overlay with MOD-2's final run.
- Suggested slide: the Results table's PR-AUC and gap rows, titled "Late rate
  helps the linear model, not the trees".

### EXP-2: full-year training data (Jul 2024 – Mar 2026)

**Question:** does training XGBoost with MOD-2's tuned settings fixed on
Jul 2024 – Mar 2026 (21 months) instead of Jul 2025 – Mar 2026 (9 months)
improve validation PR-AUC and F1? Only the training window changes.

#### Setup

**What changes.** Train gains Jul 2024 – Jun 2025: 6,947,604 rows, so train
grows from 5,105,534 to 12,053,138 rows (2.36×). That changes two things at
once: **volume** (more rows) and **seasonal coverage** (train now contains
April–June, the months of validation and test, which the current train never
sees). It also adds older data, whose schedules and carriers differ more from
2026's, which could pull the model away from April 2026. This run measures the
combined effect. A flat result means "no net gain", not "no effect": a gain
from volume or season could be cancelled by a loss from older data.

**Run B** (added after the full-year run lost to MOD-2) adds only Apr–Jun 2025
(1,774,776 rows) to the current train: recent and in season, a third of the
full year's added rows. MOD-2 → B isolates recent seasonal data; B → full year
isolates the older months Jul 2024 – Mar 2025. A **run C** (a volume-matched
older quarter, e.g. + Jul–Sep 2024) would separate data age from volume; not
run (Future work).

**What stays fixed.** Validation (Apr 2026, 591,206 rows) and test
(May–Jun 2026, 1,199,540 rows) are the same rows; the instance filter
(completed, non-diverted, labeled flights) and the cleaning rule
(`CRSElapsedTime > 0`); the split dates; the 13 base features (no `Month`, no
`late_rate`); `encode()`; the hyperparameters
(`--params reports/xgboost/mod2_metrics.json`, no `--set`); the early-stopping
**rule** (validation log loss, patience 50, cap 6,000 trees); and the threshold
**rule** (max F1 on validation). The tree count and the threshold value are
chosen again by those rules: each model stops where its own validation curve
flattens and is scored at its own best cutoff. Forcing EXP-2 to MOD-2's 345
trees or 0.229 threshold would handicap it. Its probabilities follow a
different train late rate (below), so a fixed cutoff would move precision and
recall for reasons unrelated to how well it ranks flights. The threshold does
not affect PR-AUC, which sweeps every cutoff.

**Compared against** `reports/xgboost/mod2_metrics.json` (W&B run `qy84vjlu`,
the only `MOD-2` run tagged `final`), on validation PR-AUC, F1,
`pr_auc_over_floor`, the train vs. validation gap, trees kept, and
`train_seconds` (single runs, so reported per round with MOD-3's timing as the
MOD-2 reference; see Caveats). The gap uses
`pr_auc_over_floor`, not raw PR-AUC, because the train floor drops from 0.2241
to 0.2146 (below). Validation's floor is unchanged (0.202), so validation
PR-AUC compares directly. Before the real run, MOD-2 is rerun with the new code
and no new flags; it must reproduce the committed JSON exactly (best iteration
344, same PR-AUC to every digit), which shows the plumbing did not change the
baseline.

**Same settings on more data.** Most of MOD-2's settings mean the same thing
at any train size: `learning_rate`, `subsample`, and `colsample_bytree` are
rates or fractions; `max_depth` (5) and `max_cat_threshold` (120) count levels
and categories, not rows; early stopping counts rounds. Two are absolute
amounts that weaken relative to the data as train grows:

- `min_child_weight` (26) is a minimum **sum of hessians** *p*(1 − *p*) per
  leaf: about 151 rows at *p* ≈ 0.22, about 550 rows in a leaf predicting 5%
  late. In the saved MOD-2 model (345 trees, 10,972 leaves) it binds on a small
  tail: minimum leaf cover exactly 26.0, 1st percentile 33.9, median 4,954;
  2.6% of leaves under 52 and 11.0% under 260. These are likely rare
  categories. With about 2.4× more rows, a rare airport clears the limit more
  easily, so the constraint is looser in effect.
- `reg_lambda` (7.47) shrinks each leaf value *w* = −*G* / (*H* + λ) by
  λ / (*H* + λ): 22% at *H* = 26, 11% at *H* = 60, 0.15% at the median leaf. It
  matters only in the same small leaves, and its pull halves when a leaf's rows
  double.

So the same settings regularize the full-year model less. That is how a fixed
penalty should behave when there is more evidence (like the α = 20 pretend
flights in EXP-4), so it is part of what more data buys, not a flaw in the
comparison. Whether that widens the gap (more category-specific leaves) or
narrows it (less noisy leaf values) is measured, not assumed: the share of
leaves under 52 is compared between the two models. The settings were tuned for
5.1M rows and are not re-tuned (Caveats). More trees are expected: with more
data, validation log loss should keep improving for longer.

**The new data.** Downloaded with
`scripts/fetch_bts_data.py --start 2024-07 --end 2025-06 --out data/raw_2024`,
a separate folder, so `data/raw` still holds the 12 files `preprocess.py`
asserts. Checked with `notebooks/EXP_2_Check_new_months.py` (read-only), which
applies the same filter and cleaning rule as `preprocess.py`:

- **Format:** all 12 files have the same 110-column header as `data/raw`, and
  the 20 `COLS` read with the same dtypes.
- **Quality:** after the filter, 0 nulls, 0 rows with `CRSElapsedTime ≤ 0`, 0
  rows breaking any PRE-4 out-of-range rule, 0 duplicates on the PRE-4 flight
  key, 0 rows dated outside their file's month. The existing cleaning rule needs
  no change.
- **Rows and late rate per month** (after filter and cleaning):

| Month | 2024-07 | 08 | 09 | 10 | 11 | 12 | 2025-01 | 02 | 03 | 04 | 05 | 06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rows (k) | 614 | 604 | 578 | 609 | 572 | 585 | 522 | 496 | 592 | 578 | 598 | 599 |
| Late rate | **.296** | .235 | .155 | **.132** | .147 | .212 | .188 | .208 | .196 | .197 | .236 | .283 |
| Cancelled (k) | **18.4** | 12.7 | 3.5 | 6.3 | 2.6 | 4.2 | 16.3 | 7.4 | 6.9 | 4.9 | 6.3 | 9.7 |

| | Rows | Late rate (floor) |
|---|---|---|
| Current train (Jul 2025 – Mar 2026) | 5,105,534 | 0.2241 |
| New months (Jul 2024 – Jun 2025) | 6,947,604 | 0.2076 |
| **Full-year train (Jul 2024 – Mar 2026)** | **12,053,138** | **0.2146** |
| Apr–Jun 2025 only (run B) | 1,774,776 | 0.2389 |

- Row counts follow the airline schedule: largest in summer (Jul 2024), smallest
  in Jan–Feb 2025 (February has 28 days). Holiday months are not the largest.
- The new months are **less** late overall (20.8% vs. 22.4%). July 2024 is the
  outlier (29.6% late and the most cancellations of any month), and Sep–Nov 2024
  were unusually on time (13–15%). Cancelled flights are removed by the instance
  filter, so the worst operational days count less in train than they were.
- April 2025 (19.7%) is close to validation's April 2026 (20.2%).

**New categories in train.** Category lists come from two places. `preprocess.py`
(`convert_types`) makes `Reporting_Airline`, `Origin`, `Dest`, the states, and the
time blocks categories **before** splitting, so their category lists hold every
value in all the months it loaded, including values that occur only in
validation or test. `encode()` then gives validation the train column's list, so
for these columns nothing becomes missing. `OriginCityMarketID` and
`DestCityMarketID` are plain integers in the parquet files and become
categories in `encode()` from train's values only, so a validation or test
value absent from train becomes missing. Adding rows to train only adds values;
nothing is dropped. The new months add these values the current train has no
rows for:

| Value | Rows in new months | Months present | In validation / test? |
|---|---|---|---|
| Airline `9E` | 97,482 | Jul–Dec 2024 only | No |
| Airports ALW, ITH, VLD (and their city markets) | ~960 | mostly 2024 | No |
| Airport CSG (city market 31150) | 476 | 2024 | **Yes, test** (PRE-6) |
| Airport LNS (city market 33032) | 0 | — | Test only, still unseen |

No new states or time blocks.

- **`9E` rows are kept.** Dropping them would add a filter MOD-2 does not have, a
  second change in a one-change experiment. The current train already has the
  same situation: airline `HA` has 39,849 rows in Jul–Dec 2025 and none from
  Jan 2026, so it is absent from validation and test, as is `9E`. The model can
  spend splits on these airlines that never pay off on validation; their rows
  still teach airport, route, and time-of-day patterns.
- **EVAL-4 side effect.** For MOD-2, test's 51 CSG/LNS rows (`Origin`) get valid
  `Origin` codes for categories with **no training rows** (the categories exist
  because `convert_types` saw the test months), while their
  `OriginCityMarketID` becomes missing (checked: 0 of 51 missing in `Origin`, 51
  of 51 in `OriginCityMarketID`). For EXP-2, CSG and its city market 31150 have
  476 training rows; LNS stays without training rows. How XGBoost routes a
  category that has a code but no training rows is not checked yet; EVAL-4
  should test it on a toy model, as MOD-3 did for LightGBM.
- **The processed files differ only in category lists.** `data/processed_fullyear`
  validation and test have the same rows and values as `data/processed`
  (compared column by column), but `Reporting_Airline`, `Origin`, and `Dest`
  list 4 more categories (9E, ALW, ITH, VLD), because `convert_types` saw the
  2024 months. `DataFrame.equals` is therefore False. It does not affect
  scoring: `encode()` replaces validation's list with train's.
- Side note for EVAL-4: `pd.Categorical(values, categories=...)` with values
  outside the list raises a pandas deprecation warning ("will raise in a future
  version"). For `Origin`/`Dest` this never happens (see above), but it does for
  test's city markets 31150 and 33032. It works on the current pandas; an
  upgrade before EVAL-4 would break it.

**Run B's splits** (`uv run scripts/preprocess.py --dataset aprjun2025`, folder
`data/processed_aprjun2025`): train 6,880,310 rows, Apr 2025 – Mar 2026, late
rate 0.2279; validation and test identical to `data/processed`, including the
category lists (`DataFrame.equals` True: Apr–Jun 2025 adds no new airline or
airport).

**Plumbing check.** With the new code and no new flags, `--params` reproduced
MOD-2 exactly (best iteration 344; `features`, `hyperparameters`, `train`, and
`validation` identical to the committed JSON) with all writes stubbed. Flag
combinations without a `RUNS` row (`--late-rate` with a non-base dataset) and a
non-base `--dataset` without `--params` stop before loading data.

#### Results

| | MOD-2 | **Run B** (+ Apr–Jun 2025) | **EXP-2** (+ Jul 2024 – Jun 2025) |
|---|---|---|---|
| Train months / rows | 9 / 5,105,534 | 12 / 6,880,310 | 21 / 12,053,138 |
| Train floor (late rate) | 0.2241 | 0.2279 | 0.2146 |
| Validation PR-AUC | **0.3317** | **0.3314** (−0.0003) | 0.3259 (−0.0058) |
| `pr_auc_over_floor` (validation) | 1.642 | 1.641 | 1.614 |
| F1 | **0.4072** | 0.4061 | 0.4044 |
| Threshold | 0.229 | 0.230 | 0.213 |
| Precision / recall | 0.297 / 0.647 | 0.295 / 0.650 | 0.290 / 0.666 |
| `pr_auc_over_floor` train vs. validation | 1.778 / 1.642 | 1.767 / 1.641 | 1.712 / 1.614 |
| Gap | 8.3% | 7.7% | 6.1% |
| Validation log loss | 0.47315 | 0.47369 | 0.47353 |
| Trees kept | 345 | 822 | 756 |
| Leaves with cover < 52 | 2.58% | — | 2.33% |
| `train_seconds` (rounds; s per round) | 40.7 median (395; 0.103) | 88.6 (872; 0.102) | 141.9 (806; 0.176) |

Source: `reports/xgboost/mod2_metrics.json` (W&B `qy84vjlu`),
`reports/xgboost/exp2_aprjun_metrics.json` (W&B `n381mehs`, `EXP-2-aprjun`, tag
`12-training-months`), `reports/xgboost/exp2_metrics.json` (W&B `pwkimhxm`,
`EXP-2`, tag `21-training-months`); MOD-2's `train_seconds` is the MOD-3 median.
Leaf cover from the saved models (`models/*.joblib`).

- **A full extra year makes validation worse.** −0.0058 PR-AUC is about half
  of what tuning gained over the default model and six times the 0.001 we
  called a tie in MOD-3. F1 drops 0.003.
- **Adding only recent April–June ties MOD-2.** −0.0003 PR-AUC is within the
  tie range. Train seeing April–June, the motivation for EXP-2, gives no
  measurable gain. The model has no `Month` input, so it cannot tell an April
  flight from any other; season can only help indirectly, through shifts in
  how airports, airlines, and hours relate to delays. EXP-3 (`Month` as a
  feature) is where seasonal data could pay off, and run B's dataset is the
  one to test it on: it has every month value and costs nothing against MOD-2.
- **So the full year's loss comes from the older months, Jul 2024 – Mar 2025**
  (B → full year: −0.0055). Extra months alone do not hurt (run B). Two
  explanations fit and this design cannot separate them: the older months'
  patterns are out of date for April 2026 (`9E`, July 2024's 29.6% late rate,
  an unusually on-time fall 2024), or they are so many (5.17M rows) that the
  recent 9 months fall from 74% of train (run B) to 42%. Run C would tell them
  apart.
- **The gap shrank (8.3% → 6.1%) because train got harder, not because the
  model generalizes better.** Train `pr_auc_over_floor` fell 1.778 → 1.712 and
  validation fell too. 21 months of mixed patterns are harder to fit; the gap
  only means something when validation does not get worse (same lesson as
  MOD-3's 4-leaf run). Weaker effective regularization did not widen it: the
  share of leaves at the `min_child_weight` limit fell (2.58% → 2.33%), so more
  rows per category mattered more than the fixed λ and `min_child_weight`
  getting relatively weaker.
- **The threshold followed the train late rate:** 0.229 → 0.213 for the full
  year (train floor 0.224 → 0.215), 0.230 for run B (0.228). Re-picking it was
  necessary; a fixed 0.229 would have flagged fewer flights for the full-year
  model.
- **More data, more trees, as expected:** 345 → 822 (run B) and 756 (full year).
  Per round, run B costs the same as MOD-2 (0.102 vs. 0.103 s) despite 1.35×
  the rows; the full year costs 1.7× per round for 2.36× the rows. Single runs
  (Caveats).

**Was the full-year model too small? Depth check.** MOD-2's settings were tuned
for 5.1M rows, so the full-year loss could mean depth-5 trees lack capacity for
12M rows (underfitting) rather than older data not carrying over. Only
`max_depth` changed, on the full-year train (`--set max_depth=6` / `7`; W&B
`EXP-2`, tag `depth-check`, not `final`):

| `max_depth` | Validation PR-AUC | F1 | `pr_auc_over_floor` train / validation | Gap | Trees |
|---|---|---|---|---|---|
| 5 (EXP-2) | 0.3259 | 0.4044 | 1.712 / 1.614 | 6.1% | 756 |
| 6 | 0.3259 | 0.4049 | 1.735 / 1.613 | 7.5% | 444 |
| 7 | 0.3246 | 0.4044 | 1.735 / 1.607 | 7.9% | 176 |

Source: W&B `pwkimhxm`, `1hmldp7f`, `o6nedod4`.

Deeper trees fit train better (1.712 → 1.735) but validation stays flat, then
drops, and the gap grows. Underfitting would have shown validation rising with
train. The extra capacity goes into patterns that do not carry over to April
2026, which points away from capacity and toward the data. A full hyperopt
search on the full year (estimated 5–8 hours for 90 trials) was therefore not
run: depth is the main capacity setting and more of it did not help. No
depth-6 run on the base data was needed, since no deeper full-year model came
close to MOD-2.

**Verdict: not adopted.** MOD-2 trained on Jul 2025 – Mar 2026 stays the tree
model for EVAL-5 and EVAL-4. This answers PRE-8: the raw data is not extended
back to 2024-07 for the project's dataset. `data/processed` is unchanged;
`data/processed_fullyear` and `data/processed_aprjun2025` exist only for EXP-2
(and a possible EXP-3 on run B's data).

#### Caveats

- **Settings not re-tuned.** The hyperparameters were tuned on the 9-month train.
  The depth check makes a capacity shortfall unlikely, but a full search on each
  dataset with an equal budget (90 trials) was not done.
- **Age vs. volume not separated** (run C not run). "The older months cost
  0.0055" is measured; *why* is inferred.
- **One window of validation.** All comparisons are on April 2026 only; early
  stopping and the threshold use it in every run, so each validation score is
  optimistic by the same mechanism. No bootstrap interval; run B's −0.0003 is
  called a tie by the MOD-3 rule, not by a test. Test (EVAL-4) scores only the
  adopted model (MOD-2).
- **Timing is single runs.** The median-of-3 alternating comparison planned in
  Setup was not done; MOD-3 showed single timings on this laptop vary by up to
  ~50%, so `train_seconds` and seconds per round above are indicative only.
- **The depth check is extra tuning on validation**, though it changed nothing
  that was adopted.
- **Test categories (for EVAL-4):** see "EVAL-4 side effect" above. MOD-2 is the
  model scored on test, so CSG/LNS keep `Origin` codes with no training rows and
  missing city markets.

#### Figures

- `reports/xgboost/figures/exp2_pr_curve.png`, `exp2_thresholds.png` (full year)
- `reports/xgboost/figures/exp2_aprjun_pr_curve.png`, `exp2_aprjun_thresholds.png` (run B)
- W&B: filter by name `EXP-2` / `EXP-2-aprjun` or tags `21-training-months`,
  `12-training-months`, `depth-check`. Overlay `val_logloss` history of
  `qy84vjlu`, `n381mehs`, and `pwkimhxm` to show all three curves flatten near
  0.4732–0.4737.
- Suggested slide: the Results table's first four rows (train months, PR-AUC,
  F1, gap), titled "More data did not help: recent spring ties, a whole older
  year hurts".

---

## EVAL-4: the one-time test score

### Setup

- **What was scored.** MOD-1 (baseline) and MOD-2 (tree model, chosen in MOD-3
  for training speed) are the final models; EXP-2 and EXP-4 adopted nothing.
  MOD-3 was scored as a reference row only, with the rule written down before
  scoring that its test result could not change which tree model is final
  (picking between two tied models on test would use test to choose).
- **Frozen before test was read.** Each model with the threshold saved in its
  `.joblib` (chosen on validation by max F1), and a SHA-256 hash of the file,
  because `models/` is gitignored and git cannot show the file is unchanged.
  The freeze list was committed in `b28d0dd` and the scoring code in `258f06e`,
  both before the test run.

| Model | Role | File | Threshold | SHA-256 (first 8) |
|---|---|---|---|---|
| MOD-1 | final | `models/mod1_logreg.joblib` | 0.2333 | `41c81efb` |
| MOD-2 | final | `models/mod2_xgboost.joblib` (W&B `qy84vjlu`) | 0.2294 | `bf3157a0` |
| MOD-3 | reference | `models/mod3_lightgbm.joblib` (W&B `3k15e6d7`) | 0.2280 | `bc243388` |

- **The scoring path** (`src/final_evaluation.py`) imports nothing from the
  model scripts, so tuning code cannot reach it and later edits to those
  scripts cannot change it. It checks the three hashes, then encodes test from
  the feature list and category lists stored inside each model
  (`get_categories()` for XGBoost, `pandas_categorical` for LightGBM; MOD-1's
  pipeline does its own one-hot encoding), predicts, and scores with
  `score_at_threshold` in `src/evaluation.py`, which takes the threshold as an
  argument and never searches for one (`evaluate()` picks the max-F1
  threshold on whatever data it gets, so calling it on test would tune on test).
- **Checked before test.** `--split val` runs the same path on validation and
  must reproduce every saved validation metric: all three matched exactly
  (difference 0, same confusion counts), and the stored category lists match
  what `encode()` builds from `train.parquet` for all 9 columns.
- **Score once.** `--split` is required (no default), and `--split test`
  refuses to run once `reports/final/eval4_metrics.json` exists. The JSON is
  written before the figures and W&B, so a later failure cannot cause a rescore.

### Results

| | MOD-1 | MOD-2 (final) | MOD-3 (reference) |
|---|---|---|---|
| PR-AUC, validation → test | 0.3154 → 0.3519 | **0.3317 → 0.3685** | 0.3308 → 0.3671 |
| `pr_auc_over_floor` | 1.562 → 1.494 (−4.4%) | **1.642 → 1.564 (−4.7%)** | 1.638 → 1.559 (−4.8%) |
| F1 | 0.3932 → 0.4364 | **0.4072 → 0.4464** | 0.4078 → 0.4468 |
| F1 ÷ `f1_floor` | 1.170 → 1.145 | 1.212 → 1.171 | 1.213 → 1.172 |
| Precision / recall (test) | 0.337 / 0.621 | 0.346 / 0.628 | 0.346 / 0.631 |
| Share of flights flagged late | 43.6% → 43.5% | 44.0% → 42.7% | 44.4% → 42.9% |
| Log loss better than no-skill | 4.7% → 5.4% | 6.0% → 6.4% | 5.9% → 6.4% |

Floor (late rate) 0.202 → 0.2355; `f1_floor` 0.336 → 0.381. Test confusion
matrix for MOD-2: 177,438 TP, 334,996 FP, 105,109 FN, 581,997 TN.

Source: `reports/final/eval4_metrics.json`.

- **Raw PR-AUC and F1 went up, and that is not the model getting better.**
  May–June has more late flights, which raises the score of any model,
  including guessing (no-skill PR-AUC = the late rate). Against each split's
  own floor, every model scored about 4–5% lower on test.
- **Most of the drop is the test period, not tuning optimism.** The untuned
  baseline (MOD-1) dropped 4.4%; the two tuned models, whose settings, tree
  count, and threshold were all chosen on validation, dropped 4.7–4.8%. Only
  the extra ~0.4 points can be blamed on validation being optimistic. The
  rest is shared: test is later (further from the training months) and covers
  May and June, months train (Jul 2025 – Mar 2026) does not contain.
- **The drop is small and partly mechanical.** `pr_auc_over_floor` cannot reach
  the same maximum on both splits (its ceiling is 1 ÷ floor: 4.95 on
  validation, 4.25 on test), and log loss relative to a no-skill model got
  slightly *better* on test for all three models. So the threshold-free
  evidence is: ranking held up roughly as well as on validation.
- **The ranking of the models held.** MOD-2 beat MOD-1 by +0.0166 PR-AUC on
  test vs. +0.0163 on validation (`pr_auc_over_floor` +4.7% vs. +5.2%), and by
  +0.010 F1. MOD-3 stayed tied with MOD-2 (−0.4% `pr_auc_over_floor`, +0.0004
  F1), so the MOD-3 decision (tie, XGBoost 3.6× faster) stands; as agreed
  before scoring, this row does not change the choice.
- **The frozen thresholds still worked.** They flagged a similar share of
  flights late as on validation, and recall stayed at 0.62–0.63 while precision
  rose with the late rate. F1 relative to its floor fell somewhat more than
  PR-AUC relative to its floor (MOD-2 −3.4%), which is the expected cost of a
  threshold fixed in advance. It was not re-tuned on test.

### Unseen test categories (CSG, LNS)

51 `Origin` and 48 `Dest` test rows (0.004% of test) use CSG or LNS, which have
no training rows. Each model handled them differently, which matches the
checks run on validation rows beforehand:

| Model | `Origin` / `Dest` | City market IDs | Effect |
|---|---|---|---|
| MOD-1 | not in its one-hot categories: all-zero columns (51 / 48 rows) | all-zero columns (51 / 48) | adds 0 to the log-odds; the median airport coefficient is −0.006, so about a typical airport |
| MOD-2 | kept as categories with 0 training rows (51 / 48), because `preprocess.py` built the lists from all months | missing (51 / 48) | a listed-but-untrained code goes the "not listed" way at each split; that matched the missing-value branch in only 32% (`Origin`) / 0% (`Dest`) of 5,000 probe rows, with at most 0.03 change in probability |
| MOD-3 | kept as categories with 0 training rows (51 / 48) | missing (51 / 48) | LightGBM sends unlisted and missing values the same way, so identical to missing |

Counts come from the `unseen` and `listed_but_untrained` blocks in the JSON. With
99 rows out of 1.2 million, they cannot move the headline scores.

### Caveats

- One test period (two months, one season). The comparison is unbiased but
  says nothing about winter or a different year.
- The MOD-2 vs. MOD-1 gap (+0.017 PR-AUC) is consistent across validation and
  test but has no confidence interval; no bootstrap was run.
- The test results are final: no model, threshold, or encoding was changed
  after scoring, and the script refuses a second run.

### Figures

- `reports/final/figures/eval4_val_vs_test.png`: `pr_auc_over_floor` per model,
  validation next to test. Suggested final results slide, titled "Test confirms
  the ranking: XGBoost beats the baseline by the same margin".
- `reports/final/figures/eval4_test_pr_curves.png`: the three test PR curves
  with each frozen threshold marked; MOD-2 and MOD-3 lie on top of each other.

---

## EVAL-5: where MOD-2 makes its errors (validation)

Validation only (Apr 2026, 591,206 flights); test is never read. Uses the
frozen MOD-2 file (hash checked) and the same encoding path as EVAL-4. Script:
`src/error_analysis.py`, run with `uv run python -m src.error_analysis`.

### Which features the model relies on

Two measures, shown side by side because they answer different questions:

- **Total gain** (`importance_type="total_gain"`): how much each feature
  improved the *training* loss, summed over all its splits, while the trees
  were built. Not `weight` (number of splits), which favors many-valued columns
  (`Origin` has 2,551 splits but a small average gain of 145; `DayOfWeek` 507
  splits at 401 each), and not `gain` (average per split), which ignores how
  often a feature is used.
- **Mean |SHAP|** on validation (`pred_contribs=True`): how far each feature
  moves the predictions on new flights, in log-odds. Checked that each flight's
  contributions plus the bias add up exactly to its predicted probability. It
  measures influence, not correctness. The two use different units, so compare
  shares.

| Feature | Total gain share (train) | Mean \|SHAP\| share (validation) |
|---|---|---|
| `ArrTimeBlk` | 20.9% | 19.4% |
| `DepTimeBlk` | 18.9% | 18.0% |
| `Reporting_Airline` | 7.1% | 12.5% |
| `DayOfWeek` | 7.3% | 10.4% |
| `Origin` | 13.2% | 10.2% |
| `Dest` | 10.8% | 9.7% |
| `DayofMonth` | 6.4% | 5.4% |
| `OriginCityMarketID` / `DestCityMarketID` | 6.3% / 5.8% | 4.7% / 4.7% |
| `CRSElapsedTime` | 0.4% | 2.3% |
| `OriginState` / `DestState` | 1.7% / 1.1% | 1.3% / 0.9% |
| `Distance` | 0.2% | 0.4% |

Source: `reports/eval5/eval5_feature_importance.csv`.

- **When the flight is scheduled matters most.** The two time blocks carry
  about 40% on both measures: later flights are more often late.
- **Where matters next, but less on new data than in training.** The airport
  and city market columns lose share on validation (`Origin` 13.2% → 10.2%,
  markets about 6% → 4.7%), while low-cardinality columns gain (airline 7.1% →
  12.5%, day of week 7.3% → 10.4%). With 358 airports, many splits fit
  training-specific patterns of a few airports; that gain does not carry over,
  consistent with MOD-2's 8.3% train–validation gap.
- **Flight length barely matters** (`Distance` and `CRSElapsedTime` under 3%
  together).

### Which groups the model ranks well and badly

Each group (an airline, an origin airport, a departure time block) is scored
on its own flights with `compute_loss`. Groups differ in late rate, so the
comparison is `pr_auc_over_floor` (PR-AUC ÷ the group's own late rate).

**Minimum group size: 1,000 validation flights.** Groups below it stay in the
CSVs (`enough_flights` = False) but are not ranked or quoted. Reason: a model
with no skill (random scores, true value 1.0) scores between 0.86 and 1.54
(5th–95th percentile) on a group of 200 flights with 30 late, by luck alone,
and 0.92–1.18 on 1,000 flights. The cutoff keeps all 13 airlines and 19 time
blocks, and 93 of 342 origin airports, which still cover 90.4% of validation
flights (the median airport has 238 April flights).

| Group | Best (`pr_auc_over_floor`) | Worst |
|---|---|---|
| Airline | WN 1.80, G4 1.63, UA 1.58 | F9 1.43, NK 1.41, MQ 1.32 |
| Origin airport | SJC 2.41, BOI 2.37, FAT 2.31 | CLT 1.36, JFK 1.34, ORD 1.24 |
| Departure block | 0001–0559 1.80, 0600–0659 1.73, 0800–0859 1.61 | 2000–2059 1.36, 1600–1659 1.33, 1800–1859 1.33 |

Source: `reports/eval5/eval5_groups_*.csv` (every group, with flights, late
rate, mean predicted probability, PR-AUC, log loss).

- **What a group's score means.** Inside one group the grouping feature is
  constant, so it cannot help: the score only measures how well the model
  ranks flights *within* that group. Every time block therefore scores below
  the model's overall 1.64.
- **Morning ranks best, evening worst.** Two reasons. The ratio's ceiling is
  1 ÷ late rate (12.2 for 0001–0559 at 0.082, 3.5 for 1800–1859 at 0.283), so
  groups with few late flights can score higher. And (a hypothesis, not
  tested) evening delays are mostly knock-on delays from earlier flights and
  afternoon weather, which the schedule-only features cannot see; the first
  flight of the day does not inherit a delay.
- **Large hubs rank worst** (ORD, JFK, CLT): many flights with similar
  schedules, delayed by congestion and weather the features do not include.
- **Where the model expects the wrong number of late flights.** Overall it
  predicts 0.227 on average against April's 0.202 (train's late rate is 0.224),
  so about +0.025 is expected everywhere. Beyond that: it **under-predicts**
  Spirit (NK: 0.279 vs. 0.346 actual), SFO (0.201 vs. 0.253), and DAL (0.239
  vs. 0.289), and **over-predicts** ANC (0.173 vs. 0.067), EWR (0.245 vs. 0.150),
  JFK (0.258 vs. 0.184), and Alaska (AS: 0.225 vs. 0.158). These groups had a
  different April than their training months.

### Representative false positives and false negatives

At the frozen threshold 0.2294: 182,807 false positives (flagged late, on
time) and 42,159 false negatives (not flagged, late). Most are close calls:
the median false positive has probability 0.304 (10th–90th percentile
0.242–0.425) and the median false negative 0.170 (0.101–0.218). There are 4×
more false positives because max-F1 picked a low threshold that flags 44% of
flights to catch 65% of the late ones.

Three of each kind were picked two ways: the most **confident** mistakes (blind
spots) and a **random** sample (seed 42, what a typical error looks like).
"Top reasons" are the three largest SHAP contributions in log-odds (+ pushes
toward late); every flight starts from a bias of −1.247 (probability 0.22).

| Kind | Flight (Apr 2026) | Probability | Top reasons |
|---|---|---|---|
| Confident FP | G4 HTS→SFB, Sun 12th, 9 pm | 0.822 | `Dest` +0.50, `Origin` +0.46, `ArrTimeBlk` +0.42 |
| Confident FP | G4 CKB→SFB, Sun 26th, 9 pm | 0.769 | `Dest` +0.40, `ArrTimeBlk` +0.37, `DayOfWeek` +0.37 |
| Confident FP | G4 SFB→HTS, Sun 26th, 6 pm | 0.753 | `ArrTimeBlk` +0.41, `DepTimeBlk` +0.36, `Origin` +0.34 |
| Confident FN | OO EKO→SLC, Tue 7th, 6 am | 0.016 | `Origin` −0.66, `DepTimeBlk` −0.53, `ArrTimeBlk` −0.38 |
| Confident FN | WN ITO→HNL, Fri 10th, 6 am | 0.022 | `DepTimeBlk` −0.76, `ArrTimeBlk` −0.50, `Origin` −0.45 |
| Confident FN | OO EKO→SLC, Fri 17th, 6 am | 0.023 | `Origin` −0.65, `DepTimeBlk` −0.58, `ArrTimeBlk` −0.40 |
| Random FP | OO ORD→PIT, Mon, 3 pm | 0.391 | `Origin` +0.35, `ArrTimeBlk` +0.13, `OriginCityMarketID` +0.12 |
| Random FP | AS GEG→SEA, Fri, 2 pm | 0.288 | `Reporting_Airline` +0.13, `DayOfWeek` +0.11, `DepTimeBlk` +0.09 |
| Random FP | WN SLC→DEN, Fri, 4 pm | 0.392 | `Dest` +0.23, `DepTimeBlk` +0.22, `ArrTimeBlk` +0.20 |
| Random FN | F9 DCA→DEN, Fri, 7 am | 0.150 | `DepTimeBlk` −0.35, `ArrTimeBlk` −0.33, `Reporting_Airline` +0.20 |
| Random FN | OO SFO→SMF, Wed, 6 pm | 0.190 | `DayOfWeek` −0.23, `ArrTimeBlk` +0.16, `DepTimeBlk` +0.16 |
| Random FN | AS SEA→SNA, Mon, 1 pm | 0.211 | `Dest` −0.22, `DayOfWeek` +0.13, `Reporting_Airline` +0.10 |

Source: `reports/eval5/eval5_error_cases.csv`.

- **Confident false positives: the model learned a real pattern, and these
  flights broke it.** Allegiant's (G4) Sunday-evening flights into Orlando
  Sanford from small airports. In training, G4's evening flights into SFB were
  late 52% of the time (2,051 flights; G4 overall 25%). Route, time, and
  Sunday all push toward late; these particular flights were on time.
- **Confident false negatives: the blind spot.** First flights of the day from
  small airports (SkyWest from Elko; Southwest Hilo→Honolulu). Everything the
  model can see says on time: EKO departures were late 3.3% of the time in
  training (429 flights) and the 0600–0659 block 10.3% (vs. 22.4% overall).
  Whatever delayed them (weather, a mechanical problem, a late crew) is not in
  the schedule features.
- **Typical errors are close calls.** The random cases sit 0.06–0.16 from the
  threshold, and no single feature dominates: small pushes from the airport,
  the time block, and the day add up to just over or under the line. These are
  the price of one threshold for every flight, not a specific blind spot.
- **The case for the presentation** (Project Description 3.4): the EKO→SLC
  6 am flight at 1.6% that arrived late, next to the G4 9 pm Sanford flight at
  82% that arrived on time. Together they show the model's limit: it predicts
  from the schedule, and delays caused on the day are invisible to it.

### Caveats

- Validation only (April); groups may rank differently in other months.
- SHAP measures how much a feature moves predictions, not whether that
  movement is correct; total gain is measured on training data.
- The cases are illustrations, not a sample large enough to estimate rates.
- The explanation for evening and hub airports is a hypothesis; no delay-cause
  or weather data was used to test it.

### Figures

- `reports/eval5/figures/eval5_feature_importance.png`: total gain share
  (train) next to mean |SHAP| share (validation) for all 13 features. Suggested
  slide: "Schedule time drives predictions; airports matter less on new data".
