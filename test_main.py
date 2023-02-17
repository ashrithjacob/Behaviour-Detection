import pandas as pd
#import pytest
from main import *

"""
def test_loss_calculation():
    path ='labels/'
    y_test = pd.csv_read(path+'y_test')
    y_pred = pd.csv_read(path+'y_pred')
    metric = loss(y_test, y_pred)
    y_diff=abs(y_test-y_pred)
    sumup =  y_diff.sum().sum()
    assert metric == sumup

"""

x, y, y_t = create_df()
print(x.head())
print(y.head())
print(y_t.head())
