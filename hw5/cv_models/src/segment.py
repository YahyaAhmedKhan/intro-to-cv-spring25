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
        print(img.shape)
        # print(img.size)
        torch_img = torch.from_numpy(cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0).permute(2, 0, 1).float()
        torch_img = self.transforms(torch_img)
        with torch.no_grad():
            result = self.model(torch.stack([torch_img], dim=0))

        output = result["out"][0]
        label_per_pixel = torch.argmax(output, 0)
        mask = self.color_per_class_bgr[label_per_pixel]
        mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
        print(mask.shape)

        new_img = cv.addWeighted(img, (1-self.alpha), mask, self.alpha, 0)

        plt_img = cv.cvtColor(new_img, cv.COLOR_BGR2RGB)
        # plt.imshow(plt_img)
        # plt.axis("off")
        # plt.show()

        return new_img