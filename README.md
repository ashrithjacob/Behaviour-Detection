## TODO

1. Choose video clip\
2. How to choose best frames and split (Below is what chat GPT3 had to say)\
   There are a few different strategies you might use to choose the best frames from a video for computer vision tasks:\

- Choose frames evenly spaced over time: This can be a good strategy if you want to get a representative sample of the video and don't have any particular events or features that you want to focus on. To do this, you can simply choose frames at regular intervals (e.g., every 10 seconds) throughout the video.

- Choose frames that contain specific events or features: If you are interested in a particular event or feature in the video (e.g., a person's face), you can use frame-level annotations or visual search algorithms to identify the frames that contain these events or features.

- Choose frames based on their visual quality: If the quality of the frames is important (e.g., if you are using them to train a machine learning model), you may want to choose the highest-quality frames. You can use image quality metrics (e.g., sharpness, contrast, etc.) to identify the best frames.

- Choose frames based on their content: If the content of the frames is important (e.g., if you are trying to identify objects in the video), you may want to choose frames that contain a diverse set of objects or scenes. You can use visual content analysis algorithms to identify frames with a wide range of content.

Ultimately, the best strategy for choosing frames will depend on the specific goals of your computer vision task.

3 - Yolo V5 for obj detection

- Changes made in detect.py --> detect_temp.py:Added exporting of vector csv for each\
  image in img folder
- How to train on custom data (in order to add new classes)
  create a folder next to yolov5 called datasets:\
  |\
  |\_datasets/\
  |\_yolov5/\
  In the datasets have the following structure:
  |\
  |\_name_of_dataset/images\
  |\_name_of_dataset/labels

  See: https://github.com/ultralytics/yolov5/issues/1071#issuecomment-1078443537 for\
   more details on contents of `image` and `labels` folder.

- To train a model on a specific dataset create a yaml file (eg: `coco128.yml`) in the
  `data/` folder \
  then run(Default batch_size =16, epochs =100):
  ```
  python3 train.py --img 640 --batch 2 --epochs 10 --data coco128.yaml  --weights yolov5s.pt
  ```
  Output seen in `runs/train/` folder.
- use CVAT to train datasets. See `labels.jpg` in `runs/train/exp<n>` folder to see
  how many of each class exists
- Custom training: https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb

  4 - Label frames (not on objects but on behaviour)
  Use the flask application built to do labelling

5 - Train on the data

- XGboost (using YOLO labels - should be computationally less expensive)
- simple custom CNN without YOLO Labels(see:https://www.youtube.com/watch?v=hraKTseOuJA&ab_channel=DigitalSreeni)
  ( see github: https://github.com/bnsreenu/python_for_microscopists/blob/master/142-multi_label_classification.py)
  Uses 4 layer conv net, followed by two dense layers and then a sigmoid with number of classification classes (in our case 3)

6 - Next use autoencoders (after making XGB work)
Read aout autoencoders

## Structure of project:

- Store all constants in `constants.py`
- Read data from labels and store in df
- Read data from yolov5
- trials.trials?

## TODO:

- make script to train find best hyperparams and run booster on those params.

## Pytest tools:

- Assertion
  Error on running pytest:
- File "<stdin>", line 1, in <module>
  File "/media/Data_linux/anaconda3/envs/dl/lib/python3.10/site-packages/pytest/**init**.py", line 5, in <module>
  from \_pytest.\_code import ExceptionInfo
  File "/media/Data_linux/anaconda3/envs/dl/lib/python3.10/site-packages/\_pytest/\_code/**init**.py", line 2, in <module>
  from .code import Code
  File "/media/Data_linux/anaconda3/envs/dl/lib/python3.10/site-packages/\_pytest/\_code/code.py", line 43, in <module>
  from \_pytest.\_io import TerminalWriter
  File "/media/Data_linux/anaconda3/envs/dl/lib/python3.10/site-packages/\_pytest/\_io/**init**.py", line 1, in <module>
  from .terminalwriter import get_terminal_width
  File "/media/Data_linux/anaconda3/envs/dl/lib/python3.10/site-packages/\_pytest/\_io/terminalwriter.py", line 10, in <module>
  from \_pytest.compat import final
  File "/media/Data_linux/anaconda3/envs/dl/lib/python3.10/site-packages/\_pytest/compat.py", line 256, in <module>
  @attr.s
  AttributeError: module 'attr' has no attribute 's'

## Objective function and hyperopt

- loss function = hamming distance used to optimize hyperparams
- on plugging into xgboost, we can't equal the loss acquired through hyperparam training
- create custom objective, hamming
