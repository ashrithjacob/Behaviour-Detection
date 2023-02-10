"""
# ==============================================================================================================================#
#-- Authors: Ashrith Jacob
#-- Date: January 26, 2022
#-- Description: Perfroms Xtreme Gradient Based Boosting of Trees for regression on custom human behaviour data
#-- Version : 1.0
#-- Revisions: None
#-- Reference Links Used:
#-- Required Tools:
#       python 3.7/3.8/3.9/3.10
#       pandas
# ==============================================================================================================================#
"""
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import constants
import pathlib
import numpy as np
import math
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from functools import reduce, partial
from datetime import datetime


"""
Functions:
- loss
- creat_df
- d_matrix
- get_regression_score
- run_hyperopt
"""


space = constants.space["param_space"]
loss = lambda y, y_pred: np.sum(abs(np.subtract(np.array(y), np.array(y_pred))))


def create_df():
    # getting all files in folder in alphabetical order
    files = sorted(os.listdir(constants.path_to_y))
    # concatenating dataframe in alphabetical order of file names
    y = pd.concat(
        (pd.read_csv(constants.path_to_y + str(f), sep=",", header=None) for f in files)
    )
    # reading csv containing object's presence/absence in each frame
    x = pd.read_csv(
        str(constants.path_to_x) + str(constants.x_csv), sep=",", header=None
    )
    return x, y


def create_csv(*files):
    for file in files:
        # get name of file
        name = [x for x in globals() if globals()[x] is file][0]
        if isinstance(file, pd.DataFrame):
            file.to_csv("labels/" + str(name) + ".csv", index=False, header=False)
        elif isinstance(file, np.ndarray):
            np.savetxt("labels/" + str(name) + ".csv", y_pred, delimiter=",")


def dump(obj):
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))


def check_loss(y_test, y_pred):
    y_diff = abs(y_test - y_pred)
    metric_val = loss(y_test, y_pred)
    sum_val = y_diff.sum().sum()
    print("Hamming loss is:", metric_val)
    if metric_val == sum_val:
        print("Hamming loss calculation checked to be correct")
    else:
        print("Hamming loss calculated to be incorrect")


def d_matrix():
    num_round = constants.number_of_trees
    watchlist_train = [(constants.space["data"]["d_train"], "train")]
    watchlist_tests = [(constants.space["data"]["d_test"], "tests")]
    print("loading data end, start to boost trees")
    boosted_tree = xgb.train(
        constants.best_param,
        constants.space["data"]["d_train"],
        num_round,
        evals=watchlist_train,
        verbose_eval=100,
    )
    """
    xgmat = xgb.DMatrix(x_test)
    y_pred = boosted_tree.predict(xgmat, strict_shape=True)
    accuracy = accuracy_score(y_test, y_pred > 0.5)
    return {"loss": -accuracy, "status": STATUS_OK}
    """
    return boosted_tree


def objective(space):
    num_round = constants.number_of_trees
    params = list(space.items())
    dtrain = constants.space["data"]["d_train"]
    watchlist_train = [(constants.space["data"]["d_train"], "train")]
    watchlist_tests = [(constants.space["data"]["d_test"], "test")]
    boosted_tree = xgb.train(
        params, dtrain, num_round, evals=watchlist_train, verbose_eval=100
    )
    y_pred = np.rint(
        boosted_tree.predict(constants.space["data"]["d_test"], strict_shape=True)
    ).astype(int)
    y_test = constants.space["data"]["y_test"]
    # TODO: Verify below section after reading paper
    return {"loss": loss(y_test, y_pred), "status": STATUS_OK}


# TODO: check this function too
def run_hyperopt(
    objective,
    space,
    max_evals,
    #early_stop,
    y_test,
    algorithm=tpe.suggest,
    verbose=True,
):
    try:
        trials = Trials()
        start_time = datetime.now()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            #early_stop_fn=no_progress_loss(early_stop),
            verbose=verbose,
        )
        processing_time = datetime.now() - start_time
        return (True, "Optimization Completed", best, processing_time)
    except Exception as e:
        print(e)
        return (False, "Optimization Failed:\n{}".format(e), None, None)


# tpe_trials = Trials()
# dump(tpe_trials)


if __name__ == "__main__":
    x, y = create_df()
    # print(x.head())
    # print(y.head())
    # split test and train
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=constants.test_split,
        random_state=constants.random_state,
        shuffle=True,
        stratify=y,
    )  # see docs on train size vs test size
    constants.space["data"]["y_test"] = y_test
    constants.space["data"]["d_train"] = xgb.DMatrix(x_train, y_train)
    constants.space["data"]["d_test"] = xgb.DMatrix(x_test, y_test)
    constants.space["data"]["d_test_feature"] = xgb.DMatrix(x_test)
    for i in constants.space["data"]:
        print(constants.space["data"][i])
    model = d_matrix()
    y_pred = (
        np.rint(
            model.predict(constants.space["data"]["d_test_feature"], strict_shape=True)
        )
    ).astype(int)
    check_loss(y_test, y_pred)
    create_csv(y_test, x_test, y_pred)
    status, message, best_param, processing_time = run_hyperopt(
        objective=objective,
        space=constants.space["param_space"],
        max_evals=1000,
        #early_stop=1000,
        y_test=y_test,
        algorithm=tpe.suggest,
        verbose=True
    )
    if not status:
        print(message)
    print("Best prams are", best_param)
    constants.best_param=best_param
"""
    trials = Trials()
    best = fmin(
        fn=objective,
        space=constants.space["param_space"],
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
        # early_stop_fn=no_progress_loss(early_stop),
        # verbose=True
    )
    print("Best", best)
"""


# for key in constants.space['data']:
#    print("Keys", key)
