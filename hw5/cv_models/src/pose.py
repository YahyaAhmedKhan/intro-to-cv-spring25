import torch
import numpy as np
import cv2 as cv
import torchvision


class PoseEstimator:
    coco_keypoints = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    coco_skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (5, 6),
        (6, 8),
        (7, 9),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]

    def __init__(self, detection_quality_threshold: float, keypoint_quality_threshold: float):
        self.weight = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=self.weight)
        self.transforms = self.weight.transforms()
        self.detection_quality_threshold = detection_quality_threshold
        self.keypoint_quality_threshold = keypoint_quality_threshold
        self.model.eval()

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        torch_img = (
            torch.from_numpy(cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0).permute(2, 0, 1).float()
        )
        torch_img = self.transforms(torch_img)

        with torch.no_grad():
            result = self.model([torch_img])[0]

        d_thresh = self.detection_quality_threshold
        k_thresh = self.keypoint_quality_threshold

        boxes = result.get("boxes")
        labels = result.get("labels")
        scores = result.get("scores")
        keypoints = result.get("keypoints")
        keypoints_scores = result.get("keypoints_scores")

        # print(boxes)
        # print(labels)
        # print(scores)
        # print(keypoints)
        # print(keypoints_scores)

        num_d = len(boxes)
        num_k = len(keypoints)
        for i in range(num_d):
          if scores[i] < d_thresh:
            continue
          label = labels[i]
          if label==1 or True:

            x, y, w, h = map(int, boxes[i])
            print(x, y, w, h)
            cv.rectangle(img, (x, y), (w, h), (0, 0, 255), 1)

            cv.putText(img, "person", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        for i in range(num_k):
          if labels[i] != 1: 
            continue
          points = keypoints[i]

          for j, k in PoseEstimator.coco_skeleton:
            x1, y1, _ = map(int, points[j])
            x2, y2, _ = map(int, points[k])
            
            if keypoints_scores[i][j] > k_thresh and keypoints_scores[i][k] > k_thresh:
              cv.circle(img, (x1, y1), 1, (0, 255, 0), 1, cv.LINE_AA) 
              cv.circle(img, (x2, y2), 1, (0, 255, 0), 1, cv.LINE_AA) 
              cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1, cv.LINE_AA)

        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        return img