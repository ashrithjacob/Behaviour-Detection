import pandas as pd
import numpy as np
import os
import constants
import pytest
from loss import *
import argparse
import warnings
import mlflow
# tests:TODO
# 1. check all files in labels_y folder are not empty
# 2.

"""
files = sorted(os.listdir(constants.path_to_y))
c = 0
for f in files:
    df = pd.read_csv(constants.path_to_y+str(f), header=None)
    print(str(f)+'\t'+ str(df.iloc[0].tolist()))
    c=c+1

print("length", c)
assert c == len(files)
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


def parse_opt():
    parser = argparse.ArgumentParser(prog="ProgramName", description="Description")
    parser.add_argument(
        "-hp", "--hyperopt", action="store_true", help="impliment hyperhopt"
    )
    parser.add_argument(
        "-bp", "--bestparam", choices=get_int(constants.best_param) ,help="choose best param"
    )  # takes a number of the best param that needs to be used
    args = parser.parse_args()
    print(args.hyperopt, args.bestparam)


def fxn():
    warnings.warn("parsing no arguments", UserWarning)
    x =0
    x = x + 2
    print(x)
    with mlflow.start_run(run_name='test3') as run:
        run_id = run.info.run_id
        mlflow.log_params({"a":2,"b":3,"c":4})
    return run_id

"""
x,y,y_full = create_df()
print("x shape", x.shape)
print("y shape", y.shape)
print("y_full shape", y_full.shape)
"""

parse_opt()
#print(get_int(constants.best_param))
rid = fxn()
"""
x ={ "a" : 2,
     "b" : 3,
     "c" : 4
    }
y = { "d" : 5}
print("X",x|y)
"""
#with mlflow.start_run(run_name = 'test1'):
with mlflow.start_run(run_id=str(rid)) as run:
    mlflow.log_metric("hamming",3)

