import cv2 as cv
import numpy as np
from pathlib import Path


class FaceDetector:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = cv.CascadeClassifier()
        self.model.load(str(model_path))

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = cv.equalizeHist(img_gray)

        raise NotImplementedError("YOUR CODE HERE. Use self.model.detectMultiScale() to detect faces.")

        return img
