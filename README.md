## TODO

- Choose video clip
- Choose best frames and split
  How?
  There are a few different strategies you might use to choose the best frames from a video for computer vision tasks:

Choose frames evenly spaced over time: This can be a good strategy if you want to get a representative sample of the video and don't have any particular events or features that you want to focus on. To do this, you can simply choose frames at regular intervals (e.g., every 10 seconds) throughout the video.

Choose frames that contain specific events or features: If you are interested in a particular event or feature in the video (e.g., a person's face), you can use frame-level annotations or visual search algorithms to identify the frames that contain these events or features.

Choose frames based on their visual quality: If the quality of the frames is important (e.g., if you are using them to train a machine learning model), you may want to choose the highest-quality frames. You can use image quality metrics (e.g., sharpness, contrast, etc.) to identify the best frames.

Choose frames based on their content: If the content of the frames is important (e.g., if you are trying to identify objects in the video), you may want to choose frames that contain a diverse set of objects or scenes. You can use visual content analysis algorithms to identify frames with a wide range of content.

Ultimately, the best strategy for choosing frames will depend on the specific goals of your computer vision task.

- Label frames (not on objects but on behaviour)
  Assign a class value to each image. How?
  try: https://github.com/Cartucho/OpenLabeling
  https://www.nyckel.com/
- Train on the data
- XGboost (using YOLO labels - should be computationally less expensive)
- simple custom CNN without YOLO Labels(see:https://www.youtube.com/watch?v=hraKTseOuJA&ab_channel=DigitalSreeni)
  ( see github: https://github.com/bnsreenu/python_for_microscopists/blob/master/142-multi_label_classification.py)
- Yolo V5 for obj detection
