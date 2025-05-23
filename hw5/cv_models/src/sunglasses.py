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
      if self.sunglasses is None or self.sunglasses.shape[2] != 4 or \
        self.sunglasses.shape[0] < 10 or self.sunglasses.shape[1] < 10:
          return img  

      result_img = img.copy()
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      faces = self.face_detect(gray)

      for face in faces:
          landmarks = self.face_pose(gray, face)
          landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

          # Draw all 68 facial landmarks and their indices
          # for i, point in enumerate(landmarks_points):
          #     cv.circle(result_img, point, radius=2, color=(0, 255, 0), thickness=-1)
          #     cv.putText(result_img, str(i), (point[0] + 3, point[1] - 3),
          #               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=1)

          face_points = [
              landmarks_points[17],  # top left
              landmarks_points[26],  # top right
              landmarks_points[46],  # bottom right
              landmarks_points[41]   # bottom left
          ]

          sunglasses_points = [
              (0, 10),
              (self.sunglasses.shape[1] - 10, 10),
              (self.sunglasses.shape[1] - 43, self.sunglasses.shape[0] - 43),
              (43, self.sunglasses.shape[0] - 43)
          ]

          M = cv.getPerspectiveTransform(
              np.array(sunglasses_points, dtype=np.float32),
              np.array(face_points, dtype=np.float32)
          )
          warped_sunglasses = cv.warpPerspective(self.sunglasses, M, (img.shape[1], img.shape[0]))

          mask = warped_sunglasses[:, :, 3] / 255.0
          mask_3channel = np.stack([mask] * 3, axis=2)
          warped_sunglasses_rgb = warped_sunglasses[:, :, :3]
          result_img = (1 - mask_3channel) * result_img + mask_3channel * warped_sunglasses_rgb

      return result_img.astype(np.uint8)
