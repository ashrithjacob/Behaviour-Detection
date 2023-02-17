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
import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from functools import reduce, partial
from datetime import datetime
from loss import *


"""
Functions:
- loss
- creat_df
- d_matrix
- get_regression_score
- run_hyperopt
"""


space = constants.space["param_space"]|constants.space["param_const"]


def create_df():
    # getting all files in folder in alphabetical order
    files = sorted(os.listdir(constants.path_to_y))
    # creating data frame with columns:'frame','food', 'media', 'transaction'
    y_full = pd.DataFrame(np.column_stack([files, pd.concat(
        (pd.read_csv(constants.path_to_y + str(f), sep=",", header=None) for f in files)
    ).values.tolist()]))
    # reading csv containing object's presence/absence in each frame
    x = pd.read_csv(
        str(constants.path_to_x) + str(constants.x_csv), sep=",", header=None
    )
    y= remove_first_col(y_full)
    return x, y, y_full


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


def init_dmatrix(x_train,x_test,y_train,y_test):
    constants.space["data"]["y_test"] = y_test
    constants.space["data"]["d_train"] = xgb.DMatrix(x_train, y_train)
    constants.space["data"]["d_test"] = xgb.DMatrix(x_test, y_test)
    constants.space["data"]["d_test_feature"] = xgb.DMatrix(x_test)


def get_frame(y_full,y_test):
    for row in y_test.index:
        row_name.append(row)
    return y_full.iloc[row_name,:] 


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
    row_name=[]
    x,y,y_full = create_df()
    # split test and train
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=constants.test_split,
        random_state=constants.random_state,
        shuffle=True,
        stratify=y
    )  # see docs on train size vs test size
    init_dmatrix(x_train,x_test,y_train, y_test)
    """
    status, message, best_param, processing_time = run_hyperopt(
        objective=objective,
        space=space,
        max_evals=100,
        #early_stop=1000,
        y_test=y_test,
        algorithm=tpe.suggest,
        verbose=True
    )
    if not status:
        print(message)
    print("Best params are", best_param)
    """
    #constants.best_param=best_param|constants.space["param_const"] # needs python 3.9+
    constants.best_param=constants.best_param4|constants.space["param_const"]
    print(constants.best_param)
    model = d_matrix()
    y_pred = (
        np.rint(
            model.predict(constants.space["data"]["d_test_feature"], strict_shape=True)
        )
    ).astype(int)
    check_loss(y_test, y_pred)
    y_full_test = get_frame(y_full, y_test)
    create_csv(y_full_test, y_pred)
    generate_metric()
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
