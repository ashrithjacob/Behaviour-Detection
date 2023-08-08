# Human action detection:

This project aims to detect human actions from a video. It incorporates an object detection CNN such as yolo as feature extraction followed by the use of gradient boosting through XGBoost to make predictions from the tabular data produced by YOLO.
To run the code:

```
python3 main.py -h
```

options:\
-h, --help show this help message and exit\
-hp, -- impliment hyperhopt\
-bp {1,2,3,4,5,6}, --bestparam {1,2,3,4,5,6}\
choose best param
-cu, --custom use custom objective function and eval metric as opposed to default\

### Hyperopt:
Library to tune hyperparameters to optimise a specific loss function. In this case, hamming loss.

### Best param:
Hyperopt returns the best parameters. In this case stored in `constants.py` as `best_param`.

### Custom:
The code can be run with XGBoost's built in objective function or by any custom defined functions:
custom functions that are already configured:
- RMSE (root mean squared error)
- Hamming
- Psudo-huber-loss
