import pandas as pd
import numpy as np
import csv
import constants


loss = lambda y, y_pred: np.sum(abs(np.subtract(np.array(y), np.array(y_pred))))
remove_first_col = lambda df:df[[df.columns[i] for i in range(len(df.columns)) if i != 0]].astype(str).astype(int)


def get_relevant_df():
    y_full_test = pd.read_csv("labels/y_full_test.csv", index_col=None, header=None)
    y_test = remove_first_col(y_full_test)
    y_pred = pd.read_csv("labels/y_pred.csv", index_col=None, header=None)
    y_diff = abs(y_test - y_pred)
    rows, cols = y_test.shape
    return y_full_test,y_test, y_pred, y_diff, rows, cols


# test:checking if Hamming loss calculated is correct
def check_loss(y_test, y_pred):
    y_diff = abs(y_test - y_pred)
    metric_val = loss(y_test, y_pred)
    sum_val = y_diff.sum().sum()
    print("Hamming loss is:", metric_val)
    if metric_val == sum_val:
        print("Hamming loss calculation checked to be correct")
    else:
        print("Hamming loss calculated to be incorrect")


def generate_metric():
    y_full_test,y_test,y_pred,y_diff,rows,cols=get_relevant_df()
    # writing "accuracy per frame" to text file
    file = open(constants.metrics, "w")
    file.write("Accuracy per frame:\n")
    for i in range(rows):
        file.write(str(y_full_test.iloc[i,0])+":")
        file.write(str(loss(y_test.iloc[i], y_pred.iloc[i])))
        file.write("\n")
    file.write("\n")
    # writing "accuracy per label" to text file
    file.write("Accuracy per label:\n")
    for i in range(cols):
        file.write(str(constants.labels[i])+":")
        file.write(str(y_diff.sum(axis=0)[i]))
        file.write("\n")
    file.write("\n")
    # writing hamming loss
    hamming = loss(y_test, y_pred)
    file.write("Total hamming loss:" + str(hamming)+"\n")
    file.write("Hamming percentage:" + str((hamming / (rows*cols)) * 100)+"\n")
    file.write("Total accuracy percentage:" + str(100-(hamming / (rows*cols)) * 100)+"\n")
    file.close()


#generate_metric()

# TODO:
# print the stratified graph and values in seperate folder
# add early stopping and best_params
# print frame name along with accuracy/frame - done

# Next:
# Coallesce many videos into one
# find best frame for a second along with timestamp
