"""
# ==============================================================================================================================#
#-- Authors: Ashrith Jacob
#-- Date: January 26, 2022
#-- Description: Perfroms Xtreme Gradient Based Boosting of Trees for regression on custom human behaviour data
#-- Version : 1.0
#-- Revisions: None
#-- Reference Links Used:
#-- Required Tools:
#       python 3.9/3.10
# ==============================================================================================================================#
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from src import constants
import numpy as np
import warnings
import mlflow
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from functools import reduce, partial
from datetime import datetime
from src.loss import *


"""
Functions:
- loss
- creat_df
- d_matrix
- get_regression_score
- run_hyperopt
"""


space = constants.param["space"] | constants.param["const"]  # needs python 3.9+

"""
creates data frame from csv files
returns: x, y, y_full
x: data frame containing object's presence/absence in each frame
y: data frame containing labels for each frame
y_full: data frame containing labels for each frame and file name
"""


def create_df():
    # getting all files in folder in alphabetical order
    files = sorted(os.listdir(constants.path_to_y))
    # creating data frame with columns:'frame','food', 'media', 'transaction'
    y_full = pd.DataFrame(
        np.column_stack(
            [
                files,
                pd.concat(
                    (
                        pd.read_csv(constants.path_to_y + str(f), sep=",", header=None)
                        for f in files
                    )
                ).values.tolist(),
            ]
        )
    )
    # reading csv containing object's presence/absence in each frame
    x = pd.read_csv(
        str(constants.path_to_x) + str(constants.x_csv), sep=",", header=None
    )
    y = remove_first_col(y_full)
    return x, y, y_full


"""
saves numpy array and dataframe to csv file
"""


def create_csv(*files):
    for file in files:
        # get name of file
        name = [x for x in globals() if globals()[x] is file][0]
        if isinstance(file, pd.DataFrame):
            file.to_csv("labels/" + str(name) + ".csv", index=False, header=False)
        elif isinstance(file, np.ndarray):
            np.savetxt("labels/" + str(name) + ".csv", y_pred, delimiter=",")


"""
display gradients and hessians
"""


def display(g, h, y_pred):
    print("y_test", constants.data["d_test"].get_label())
    print("Y_pred", y_pred.flatten())
    print("G,H", g, h)
    print("G,H shapes", g.shape, h.shape)
    print("G,H types", g.dtype, h.dtype)


"""
create dmatrix for xgboost
"""


def init_dmatrix(x_train, x_test, y_train, y_test):
    constants.data["y_test"] = y_test
    constants.data["d_train"] = xgb.DMatrix(x_train, y_train)
    constants.data["d_test"] = xgb.DMatrix(x_test, y_test)
    constants.data["d_test_feature"] = xgb.DMatrix(x_test)


"""
get frame_name from y_full for only the test data
"""


def get_frame(y_full, y_test):
    row_name = []
    for row in y_test.index:
        row_name.append(row)
    return y_full.iloc[row_name, :]


"""
perform xgboost regression
returns: boosted_tree
"""


def d_matrix():
    num_round = constants.number_of_trees
    watchlist = [
        (constants.data["d_train"], "train"),
        (constants.data["d_test"], "tests"),
    ]
    print("loading data end, start to boost trees")
    boosted_tree = xgb.train(
        constants.best_param[0],
        constants.data["d_train"],
        num_round,
        evals=watchlist,
        verbose_eval=100,
    )
    return boosted_tree


"""
objective function for hyperopt
"""


def objective(space):
    num_round = constants.number_of_trees
    params = list(space.items())
    dtrain = constants.data["d_train"]
    watchlist_train = [(constants.data["d_train"], "train")]
    watchlist_tests = [(constants.data["d_test"], "test")]
    boosted_tree = xgb.train(
        params, dtrain, num_round, evals=watchlist_train, verbose_eval=100
    )
    y_pred = np.rint(
        boosted_tree.predict(constants.data["d_test"], strict_shape=True)
    ).astype(int)
    y_test = constants.data["y_test"]
    return {"loss": loss(y_test, y_pred), "status": STATUS_OK}


""""
use hyperopt to find best parameters
"""


def run_hyperopt(
    objective,
    space,
    max_evals,
    early_stop,
    algorithm=tpe.suggest,
    verbose=True,
):
    try:
        trials = Trials()
        start_time = datetime.now()
        best = fmin(
            fn=objective,
            space=space,
            algo=algorithm,
            max_evals=max_evals,
            trials=trials,
            early_stop_fn=no_progress_loss(early_stop),
            verbose=verbose,
        )
        processing_time = datetime.now() - start_time
        return (True, "Optimization Completed", best, processing_time)
    except Exception as e:
        print(e)
        return (False, "Optimization Failed:\n{}".format(e), None, None)


"""
parse command line arguments
"""


def parse_opt():
    parser = argparse.ArgumentParser(
        prog="XGB with hyperopt",
        description="Running XGBoost on yolo object detection data and performing hyperparam tuning as well with it",
    )
    parser.add_argument(
        "-hp", "--hyperopt", action="store_true", help="impliment hyperhopt"
    )
    parser.add_argument(
        "-bp",
        "--bestparam",
        choices=get_int(constants.best_param),
        help="choose best param",
    )  # takes a number of the best param that needs to be used
    parser.add_argument(
        "-cu",
        "--custom",
        action="store_true",
        help="use custom objective function and eval metric",
    )
    parser.add_argument(
        "-gr",
        "--gradient",
        action="store_true",
        help="display gradient and hessian",
    )
    args = parser.parse_args()
    return args.hyperopt, args.bestparam, args.custom, args.gradient


if __name__ == "__main__":
    x, y, y_full = create_df()
    # split test and train
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=constants.test_split,
        random_state=constants.random_state,
        shuffle=True,
        stratify=y,
    )
    # create a D_Matrix represetation for required data
    init_dmatrix(x_train, x_test, y_train, y_test)
    # get parser
    hyperopt_arg, best_param_arg, custom_arg, disp_gradient = parse_opt()
    if hyperopt_arg:
        status, message, best, processing_time = run_hyperopt(
            objective=objective,
            space=space,
            max_evals=100,
            early_stop=100,
            algorithm=tpe.suggest,
            verbose=True,
        )
        if not status:
            print(message)
        print("Best params are", best)
        # create best_params:
        constants.best_param[0] = space.copy()
        constants.best_param[0].update(best)
    if best_param_arg != None:
        constants.best_param[0] = (
            constants.param["const"] | constants.best_param[int(best_param_arg)]
        )
    else:
        warnings.warn(
            "parsing no arguments, proceeding to use 'best_param: 1'", UserWarning
        )
        constants.best_param[0] = constants.param["const"] | constants.best_param[1]
    if custom_arg:
        constants.best_param[0] = constants.best_param[0] | constants.param["custom"]
        run_name = (
            constants.best_param[0]["obj"].__name__
            + "_"
            + str(constants.best_param[0]["feval"].__name__)
        )
    else:
        constants.best_param[0] = constants.best_param[0] | constants.param["default"]
        run_name = (
            constants.best_param[0]["objective"]
            + "_"
            + constants.best_param[0]["eval_metric"]
        )
    # log params
    with mlflow.start_run(run_name=str(run_name)) as run:
        run_id = run.info.run_id
        mlflow.log_params(constants.best_param[0])
    # run model:
    model = d_matrix()
    # get predictions:
    # raw
    y_pred1 = model.predict(constants.data["d_test_feature"], strict_shape=True)
    # rounded
    y_pred = (
        model.predict(constants.data["d_test_feature"], strict_shape=True) > 0.5
    ).astype(int)
    if disp_gradient:
        # gradient and hessian (set to user to print)
        g, h = custom_rmse(y_pred, constants.data["d_test"])
        display(g, h, y_pred1)
    # hamming loss
    print("Hamming loss is:", loss(y_test, y_pred))
    # get y_test df with image name
    y_full_test = get_frame(y_full, y_test)
    # write parsed args to csv
    create_csv(y_full_test, y_pred)
    # generate 'labels/metrics.txt'
    generate_metric(run_id)

# TODO:
# 1. early stop function run_hyperopt - DONE
# 2. pytest - for each function - change './\
# 3. Read on how to get hamming loss
# 4. Save the stratified graph image
# 5. Coallesce many videos into one
# 6. Find best frame for a second along with timestamp
# 7. train on more data
