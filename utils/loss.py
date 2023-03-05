import pandas as pd
import numpy as np
import csv
from utils import constants
import xgboost as xgb
import mlflow
from sklearn.metrics import classification_report 

loss = lambda y, y_pred: np.sum(abs(np.subtract(np.array(y), np.array(y_pred))))
remove_first_col = (
    lambda df: df[[df.columns[i] for i in range(len(df.columns)) if i != 0]]
    .astype(str)
    .astype(int)
)


def custom_rmse(y_pred: np.ndarray, y_test: xgb.DMatrix) -> np.ndarray:
    r, c = y_pred.shape
    gradient = 2.0 * (y_pred.flatten() - y_test.get_label())
    # gradient = np.sum(2.0*(y_pred - np.reshape(y_test.get_label(),(r,c))), axis=1)
    hessian = np.repeat(2, r * c)
    # hessian = np.repeat(2,r*c)
    return gradient, hessian


def rmse_eval(y_pred: np.ndarray, y_test: xgb.DMatrix) -> tuple[str, float]:
    return "custom-rmse", float(
        np.sqrt(
            (np.sum(np.power((y_pred.flatten() - y_test.get_label()), 2)))
            / y_test.num_row()
        )
    )


def custom_hamming(y_pred: np.ndarray, y_test: xgb.DMatrix) -> np.ndarray:
    r, c = y_pred.shape
    gradient = np.repeat(1, r * c) - 2.0 * (y_test.get_label())
    # gradient = np.sum(2.0*(y_pred - np.reshape(y_test.get_label(),(r,c))), axis=1)
    hessian = np.repeat(0, r * c)
    # hessian = np.repeat(2,r*c)
    return gradient, hessian


def hamming_eval(y_pred: np.ndarray, y_test: xgb.DMatrix) -> tuple[str, float]:
    r, c = y_pred.shape
    y_t = y_test.get_label()
    y_p = y_pred.flatten()
    return "custom-hamming", float(
        np.sum(np.multiply(y_t, (1 - y_p)) + np.multiply(y_p, 1 - y_t))
    )


def pseudo_huber_loss(y_pred: np.ndarray, y_test: xgb.DMatrix) -> np.ndarray:
    d = y_pred.flatten() - y_test.get_label()
    delta = 1
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    gradient = d / scale_sqrt
    hessian = (1 / scale) / scale_sqrt
    return gradient, hessian


def pseudo_huber_loss_eval(
    y_pred: np.ndarray, y_test: xgb.DMatrix
) -> tuple[str, float]:
    d = y_pred.flatten() - y_test.get_label()
    delta = 1
    return "psuedo-huber-loss", float(
        np.sum(delta**2 * np.sqrt(1 + (d / delta) ** 2) - 1)
    )


def get_relevant_df():
    y_full_test = pd.read_csv("labels/y_full_test.csv", index_col=None, header=None)
    y_test = remove_first_col(y_full_test)
    y_pred = pd.read_csv("labels/y_pred.csv", index_col=None, header=None)
    y_diff = abs(y_test - y_pred)
    rows, cols = y_test.shape
    return y_full_test, y_test, y_pred, y_diff, rows, cols


#return the integer values of dict keys
def get_int(dict):
    l = [val for val in list(dict.keys()) if isinstance(val, (int, float))]
    l.remove(0)
    return [str(l[i]) for i in range(len(l))]


def generate_metric(run_id):
    y_full_test, y_test, y_pred, y_diff, rows, cols = get_relevant_df()
    # classification report generation
    y_t = y_test.to_numpy()
    y_p = y_pred.to_numpy()
    report = classification_report(y_t, y_p, target_names=constants.labels)
    report_dict = classification_report(y_t, y_p, output_dict=True, target_names=constants.labels)
    # writing "accuracy per frame" to text file
    file = open(constants.metrics, "w")
    file.write("Accuracy per frame:\n")
    for i in range(rows):
        file.write(str(y_full_test.iloc[i, 0]) + ":")
        file.write(str(loss(y_test.iloc[i], y_pred.iloc[i])))
        file.write("\n")
    file.write("\n")
    # writing "accuracy per label" to text file
    file.write("Accuracy per label:\n")
    for i in range(cols):
        file.write(str(constants.labels[i]) + ":")
        file.write(str(y_diff.sum(axis=0)[i]))
        file.write("\n")
    file.write("\n")
    # writing hamming loss
    hamming = loss(y_test, y_pred)
    file.write("Total hamming loss:" + str(hamming) + "\n")
    file.write("Hamming percentage:" + str((hamming / (rows * cols)) * 100) + "\n")
    file.write(
        "Total accuracy percentage:" + str(100 - (hamming / (rows * cols)) * 100) + "\n"
    )
    file.write("\n"+"Classification Report:\n")
    file.write(report)
    file.close()
    # log metric
    with mlflow.start_run(run_id = str(run_id)) as run:
        mlflow.log_metric("hamming",hamming)
        mlflow.log_metric("hamming percentage",(hamming / (rows * cols)) * 100)
        mlflow.log_metric("accuracy percentage", 100 - (hamming / (rows * cols)) * 100)
        mlflow.log_metric("micro precision", report_dict['micro avg']['precision'])
        mlflow.log_metric("micro recall", report_dict['micro avg']['recall'])
        mlflow.log_metric("micro F1 score", report_dict['micro avg']['f1-score'])

# generate_metric()
# TODO:
# add early stopping and best_params
# print frame name along with accuracy/frame - done

# Next:
