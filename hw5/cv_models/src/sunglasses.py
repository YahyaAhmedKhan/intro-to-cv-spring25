import numpy as np
import cv2 as cv
import dlib
import requests
import bz2
import os
import shutil


class SunglassesAnnotator:
    DLIB_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    KEYPOINT_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

    def __init__(self, sunglasses_path: str, keypoint_model_path: str = "download"):
        self.face_detect = dlib.get_frontal_face_detector()

        if keypoint_model_path == "download":
            if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
                self.download_dlib_predictor()
            self.face_pose = dlib.shape_predictor(SunglassesAnnotator.KEYPOINT_MODEL_PATH)
        else:
            self.face_pose = dlib.shape_predictor(keypoint_model_path)

        # Read the sunglasses image with alpha channel
        self.sunglasses = cv.imread(sunglasses_path, cv.IMREAD_UNCHANGED)

    @staticmethod
    def download_dlib_predictor():
        # Download the Dlib predictor model
        response = requests.get(SunglassesAnnotator.DLIB_PREDICTOR_URL)
        if response.status_code != 200:
            raise ValueError("Failed to download the Dlib predictor model")
        with open("shape_predictor_68_face_landmarks.dat.bz2", "wb") as f:
            f.write(response.content)
        with open(SunglassesAnnotator.KEYPOINT_MODEL_PATH, "wb") as f:
            with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", "rb") as bz2_f:
                shutil.copyfileobj(bz2_f, f)

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("YOUR CODE HERE. OPTIONAL / EXTRA CREDIT.")
