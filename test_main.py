import pandas as pd
import numpy as np
import os
import constants
import pytest
from loss import *
from main import *
import argparse
import warnings
import mlflow
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.datasets import make_multilabel_classification

def XGB_multilabel():
    X, y = make_multilabel_classification(
        n_samples=32, n_classes=5, n_labels=3, random_state=0
    )
    clf = xgb.XGBClassifier(tree_method="hist")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), y)


# tests:TODO
# 1. check all files in labels_y folder are not empty
# 2. 


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


#parse_opt()
#print(get_int(constants.best_param))
rid = fxn()


#with mlflow.start_run(run_name = 'test1'):
with mlflow.start_run(run_id=str(rid)) as run:
    mlflow.log_metric("hamming",3)


def metric_test(y_test, y_pred):
    target_names = ['class 0', 'class 1', 'class 2']
    x = classification_report(y_test, y_pred, output_dict=True, target_names=target_names)
    precision,recall,fscore,support=score(y_test,y_pred,average='macro')
    print ('Precision : {}'.format(precision))
    print ('Recall    : {}'.format(recall))
    print ('F-score   : {}'.format(fscore))
    print ('Support   : {}'.format(support))
    print('x', x['class 0']['precision'])
    return x

def metric_test_main(y_test,y_pred):
    target_names = ['class 0', 'class 1', 'class 2']
    y_t = y_test.to_numpy()
    y_p = y_pred.to_numpy()
    print(y_t)
    print(y_p)
    report = classification_report(y_t, y_p, target_names=constants.labels)
    report_dict = classification_report(y_t, y_p, output_dict=True, target_names=constants.labels)
    print(report)

def df_to_np():
    y_pred = pd.read_csv('pred_csv.csv', header=None, index_col=None)
    y_test = pd.read_csv('test_csv.csv', header=None, index_col=None)
    y1=y_pred.to_numpy()
    y2=y_test.to_numpy()
    print(np.transpose(y2))
    print(y1)
    return y2,y1

y_full_test, y_test, y_pred, y_diff, rows, cols = get_relevant_df()
#yt,yp=df_to_np()
#print(metric_test(yt,yp))
#print(y_test.shape, y_pred.shape)
print(metric_test_main(y_test,y_pred))
