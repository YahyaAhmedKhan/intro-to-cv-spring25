import torch
import numpy as np
import cv2 as cv
from ultralytics.utils.plotting import Colors


class YOLOObjectDetector:

    def __init__(self, detection_quality_threshold: float):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.colors = Colors()
        self.detection_quality_threshold = detection_quality_threshold
        self.model.eval()

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        self.model.conf = self.detection_quality_threshold
        with torch.no_grad():
            # The YOLO model includes built-in preprocessing that can accept a list of
            # OpenCV-style images directly, as long as we do BGR to RGB conversion
            result = self.model([cv.cvtColor(img, cv.COLOR_BGR2RGB)])

        raise NotImplementedError("YOUR CODE HERE. Read the docs / use your debugger to inspect what is returned by the model")

        return img
