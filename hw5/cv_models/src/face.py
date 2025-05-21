from csv import Error
import cv2 as cv
import numpy as np
from pathlib import Path

from scipy.datasets import face


class FaceDetector:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = cv.CascadeClassifier()
        self.model.load(str(model_path))

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = cv.equalizeHist(img_gray)
        
        face_cascade = self.model
        
        faces = face_cascade.detectMultiScale(img_gray)
        
        for x, y, w, h in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            
        

        # cv.imshow("Faces", img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        return img
