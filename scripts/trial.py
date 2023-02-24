import numpy as np
import pandas as pd
import argparse
#import xgboost
import pytest
import htmlmin
#from hyperopt import hp

def square(x):
    x =np.array(x)
    return np.power(x,2)

def cube(x):
    x = np.array(x)
    return np.power(x,3)