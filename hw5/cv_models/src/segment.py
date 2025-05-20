import torch
import numpy as np
import cv2 as cv
import torchvision
from matplotlib import cm


class SemanticSegmenter:
    def __init__(self, alpha=0.8):
        self.weight = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        self.model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=self.weight)
        self.transforms = self.weight.transforms()
        self.alpha = alpha
        self.model.eval()

        self.color_per_class_bgr = np.concatenate([
            np.array([[0, 0, 0]]),  # background
            np.array(cm.get_cmap('tab20').colors)[:, ::-1] * 255
        ], axis=0).astype(np.uint8)

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        torch_img = torch.from_numpy(cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0).permute(2, 0, 1).float()
        torch_img = self.transforms(torch_img)
        with torch.no_grad():
            result = self.model(torch.stack([torch_img], dim=0))

        raise NotImplementedError("YOUR CODE HERE. Read the docs / use your debugger to inspect what is returned by the model")
