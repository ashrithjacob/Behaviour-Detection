import os
import pytest
import pandas as pd
import numpy as np
import main
from unittest.mock import MagicMock
from utils import constants as c

pred_test = MagicMock(return_value =(np.array([[3,1,2],[4,2,7],[5,1,2]]),pd.DataFrame([[4,2,2],[3,2,1],[4,5,4]], index=None)))

@pytest.fixture
def get_y_folder():
    return sorted(os.listdir(c.path_to_y))

def test_create_df_check_number_of_files(get_y_folder):
    assert len(get_y_folder) == 1000

def test_create_df_check_not_empty(get_y_folder):
    for file in get_y_folder:
        assert pd.read_csv(c.BASE_DIR + '/'+ c.path_to_y+'/'+file, header=None).empty == False

def test_create_df_check_shape(get_y_folder):
    for file in get_y_folder:
        assert pd.read_csv(c.BASE_DIR + '/'+ c.path_to_y+'/'+file, header=None).shape == (1,len(c.labels))

def test_create_df_check_vals(get_y_folder):
    x,y,y_full = main.create_df()
    vals =[0,1]
    r = len(get_y_folder)
    assert x.shape == (r,80)
    assert y.shape == (r,len(c.labels))
    assert y_full.shape == (r,len(c.labels)+1)
    assert False not in y.isin(vals).values

def test_main_check_loss():
    y_pred,y_test = pred_test()
    y_diff = abs(y_test - y_pred)
    assert c.loss(y_test, y_pred) == y_diff.sum().sum()
