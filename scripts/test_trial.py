import numpy as np
import pandas as pd
import argparse
import pytest
import htmlmin
from trial import square, cube
x =[2,4,5]
import os
def test_current():
    print("env path:",os.getenv('PYTEST_CURRENT_TEST'))

def test_square():
    y = square(x)
    for i in range(len(x)):
        assert y[i] == x[i] ** 2

def test_cube():
    y = cube(x)
    for i in range(len(x)):
        assert y[i] == x[i] ** 3

