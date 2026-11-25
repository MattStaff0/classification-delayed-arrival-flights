import numpy as np
from sklearn.model_selection import KFold


# constants
ALPHA = 20
K = 5
RANDOM_STATE = 42
KEYS = ["Origin", "DepTimeBlk"]

def rate_table(X, y):
    table = y.groupby([X[col] for col in KEYS], observed=True).agg(["sum", "size"])
    return table, y.mean()

def look_up(table, overall_rate, X):
    counts = X[KEYS].join(table, on=KEYS)

    # unseen flights get 0 late and 0 flights 
    late = counts["sum"].fillna(0).to_numpy()
    flights = counts["size"].fillna(0).to_numpy()

    # smoothing: blend the key's counts with ALPHA flights at the overall rate of the rows the table was built from
    # few flights -> close to that overall rate, many flights -> close to the key's own rate, 0 flights -> exactly the overall rate
    return (late + ALPHA * overall_rate) / (flights + ALPHA)



def add_late_rate_features(training, training_y, validation):
    # assign the folds
    folds = KFold(shuffle=True, n_splits=K, random_state=RANDOM_STATE)

    # one slot per training row, each pass in the loop will fill in its own fold's blanks
    train_late_rate = np.full(len(training), np.nan)

    # loop over the folds
    for other_folds, this_fold in folds.split(training):
        table, overall_rate = rate_table(training.iloc[other_folds], training_y.iloc[other_folds])
        train_late_rate[this_fold] = look_up(table, overall_rate, training.iloc[this_fold])

    # assert every row got a value (each row is in exactly one fold)
    assert not np.isnan(train_late_rate).any()

    # for validation it uses the data from all of the training data, no folds 
    table, overall_rate = rate_table(training, training_y)
    val_late_rate = look_up(table, overall_rate, validation)

    return train_late_rate, val_late_rate

