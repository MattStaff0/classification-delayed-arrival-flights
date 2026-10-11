# Open Issues

**Project:** _Pickaxe and Pandas_

---

_**What this file is.** Every question about the project that you cannot answer yet, in one place, with the name of the person who can answer it_

_**Why it exists.** A draft specification with confident guesses in the gaps is more dangerous than one with holes in it, because nobody can tell the guesses from the facts. Writing "we do not know" is not an admission of failure in week 3, it is the correct state. What fails is knowing and not writing it down._

**Where entries come from.**

- Entries are made when there is something that needs to get done in the project and it is an open task.

**How they leave.**

- Entries leave once they have an opened PR on github that has been merged into the main branch. Furthermore, after changes are made there should be closed in this file with an explaination of what the task / open issue contributed to.

_**Identifiers here are numbers**, `OI-1` upward, and that is deliberate. Numbers are fine for a list that only ever grows at the bottom and gets cited lightly._

## Current Identifiers

`PRE-X` - data preprocessing
`MOD-X` - Model creation
`EXP-X` - Anything relating to an experiment (running a pipeline for training a model)
`EVAL-X` - Model Evaluation

**Note on New Identifiers**:

- new identifiers can be placed here if this document is updated. A new section should be created for this

# Current Open Issues

## Example Template

- Each open issue MUST FOLLOW THIS FORMAT

### Open

| ID   | Question                                                                                                                                              | Why it matters                                                              | Who can answer                                              | Raised         |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------- | -------------- |
| OI-1 | _[The question, phrased so it has a definite answer. Not "tell us about reporting" but "who is allowed to see another student's evaluation scores?"]_ | _[What we get wrong if we guess. Name the artifact or decision it blocks.]_ | _[Client, instructor, a specific user, or the team itself]_ | _[YYYY-MM-DD]_ |

### Resolved

| ID   | Question                   | Answer                | Answered by | Date           | Filed in                                                    |
| ---- | -------------------------- | --------------------- | ----------- | -------------- | ----------------------------------------------------------- |
| OI-0 | _[Example row, delete it]_ | _[What we were told]_ | _[Name]_    | _[YYYY-MM-DD]_ | _[`BO-grading-time`, or the glossary term, or the section]_ |

## Data Preprocessing - PRE-X

### Open

| ID    | Question                                                                                                                                                                                                                                                                                                          | Why it matters                                                                                                                                                                                                       | Who can answer                 | Raised     |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ---------- |
| PRE-1 | Which exact columns go into `data/processed`? Current tentative list is `COLS` in `scripts/utils.py`. Candidates beyond it: `DayofMonth`, `DepTimeBlk`/`ArrTimeBlk`, `DistanceGroup`, `Flight_Number_Reporting_Airline`. Every kept column must be known at booking time (check each against `docs/readme.html`). | Blocks `scripts/preprocess.py`. A post-departure column slipping in (e.g. `DepDelay`, `TaxiOut`) is target leakage and invalidates every result after it.                                                            | Team                           | 2026-10-11 |
| PRE-2 | Do we drop `Month` as a feature? Train is Jul–Mar, so validation (Apr) and test (May–Jun) contain only month values the model never saw in training.                                                                                                                                                              | The proposal lists month as a feature. As one-hot it becomes all-zeros on val/test; as a number the model extrapolates. Dropping it is a change from the approved proposal and must be stated in the midterm report. | Team                           | 2026-10-11 |
| PRE-3 | How do we represent scheduled times? `CRSDepTime`/`CRSArrTime` are HHMM integers (1259 → 1300 looks like +41). Options: hour 0–23, minutes since midnight, or the BTS `DepTimeBlk`/`ArrTimeBlk` columns.                                                                                                          | Decides the derived columns written in preprocessing. The proposal describes the input as "departure hour", so hour is the default unless the team decides otherwise.                                                | Team                           | 2026-10-11 |
| PRE-4 | After filtering (`Cancelled == 0`, `Diverted == 0`, `ArrDel15` not null), how many missing / out-of-range / duplicate values remain in the kept columns, and do we drop or impute them?                                                                                                                           | Not measured yet. Logistic regression fails on NaNs; the midterm report requires data-quality findings and missingness analysis.                                                                                     | Team (measure with a script)   | 2026-10-11 |
| PRE-5 | Do the processed splits reproduce the proposal's numbers: 6,896,283 usable rows, 22.4% late overall, 0.236 late rate on test (the PR-AUC floor)?                                                                                                                                                                  | If they don't match, either the filter or the proposal is wrong, and the PR-AUC floor we compare against is wrong too.                                                                                               | Team (run `preprocess.py`)     | 2026-10-11 |
| PRE-6 | How many val/test rows have an `Origin`, `Dest`, or `Reporting_Airline` value that never appears in train?                                                                                                                                                                                                        | Determines how the model pipeline must handle unseen categories (e.g. `handle_unknown="ignore"`), and is a data-understanding finding for the report.                                                                | Team (measure after splitting) | 2026-10-11 |
| PRE-7 | What format and layout do processed files use? Proposed: `data/processed/{train,val,test}.parquet`, one file per split, split on `FlightDate`.                                                                                                                                                                    | Every MOD/EXP script reads these files. `data/` is gitignored, so the files must be fully reproducible from `preprocess.py` — the README needs the command.                                                          | Team                           | 2026-10-11 |

### Resolved

| ID   | Question                   | Answer                | Answered by | Date           | Filed in                                                    |
| ---- | -------------------------- | --------------------- | ----------- | -------------- | ----------------------------------------------------------- |
| OI-0 | _[Example row, delete it]_ | _[What we were told]_ | _[Name]_    | _[YYYY-MM-DD]_ | _[`BO-grading-time`, or the glossary term, or the section]_ |
