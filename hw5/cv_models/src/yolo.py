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
            print(result)
            # print(np.array(result.xyxy[0]).shape)
            preds = np.array(result.xyxy[0])
            for p in preds:
              x1, y1, x2, y2, prob, cls = map(int, tuple(p))
              cv.rectangle(img, (x1, y1), (x2, y2), self.colors(cls), 1)
              # cv.rectangle(img, (x1, y1), (x2, y2), self.colors(cls), 1)
              label = self.model.names[cls]
              cv.putText(img, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        return img
