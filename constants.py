import numpy as np
from hyperopt import hp

# CONSTANTS:
path_to_y = "labels/labels_y/"
path_to_x = "labels/labels_x/"
x_csv = "image_vector.csv"
metrics = "labels/metrics.txt"
labels = ["food", "media", "transactions"]
test_split = 0.2
random_state = 42
number_of_trees = 10000
space = {
    "param_space": {
        "learning_rate": hp.quniform("learning_rate", 0.1, 0.6, 0.1),
        "max_depth": hp.randint("max_depth", 3, 10),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.1),
        "gamma": hp.quniform("gamma", 0, 1, 0.05),
        "reg_lambda": hp.choice("reg_lambda", np.arange(1, 4, dtype=int)),
    },
    "param_const": {
        "alpha": 0,
        #'max_bin': max_bin,
        "n_estimators": number_of_trees,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        #'tree_method': tree_method,
        "eval_metric": "rmse",
        "silent": 1,
        "nthread": -1,
        #'seed': seed_value,
        #'gpu_id': gpu_id
    },
    "data": {},
}
best_param = {}
best_param2 = {
    "colsample_bytree": 0.5,
    "gamma": 0.6000000000000001,
    "learning_rate": 0.4,
    "max_depth": 3,
    "reg_lambda": 1,
    "subsample": 0.55,
}
best_param3 = {
    "colsample_bytree": 0.7000000000000001,
    "gamma": 0.9500000000000001,
    "learning_rate": 0.30000000000000004,
    "max_depth": 4,
    "reg_lambda": 0,
    "subsample": 0.8,
}
best_param4 = {
        "colsample_bytree": 0.9,
        "gamma": 0.7000000000000001,
        "learning_rate": 0.6000000000000001,
        "max_depth": 4,
        "reg_lambda": 2,
        "subsample": 0.5,
    }
best_param1 = {
    "learning_rate": 0.3,
    "max_depth": 3,
    "subsample": 0.5,
    "colsample_bytree": 0.75,
    "gamma": 0,
    "reg_lambda": 0,
    "alpha": 0,
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eval_metric": "rmse",
    "silent": 1,
    "nthread": -1,
}
