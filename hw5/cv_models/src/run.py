import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
from face import FaceDetector
from pose import PoseEstimator
from yolo import YOLOObjectDetector
from segment import SemanticSegmenter
from sunglasses import SunglassesAnnotator
from typing import Protocol
import time


class DetectAndDrawModel(Protocol):
    def detect_and_draw(self, img: np.ndarray) -> np.ndarray: ...


def run_single_image(img_path: Path, model: DetectAndDrawModel, output_path: Path):
    img = cv.imread(str(img_path))
    img_out = model.detect_and_draw(img)
    cv.imwrite(str(output_path), img_out)


def run_live(model: DetectAndDrawModel, camera: int = 0, resize: float = 0.5):
    cam = cv.VideoCapture(camera)
    last_frame_time = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        frame = cv.resize(frame, dsize=None, fx=resize, fy=resize)
        frame = model.detect_and_draw(frame)

        # Calculate FPS
        current_time = time.time()
        fps, last_frame_time = 1 / (current_time - last_frame_time), current_time
        cv.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imshow(f"{model.__class__.__name__} (press Q to quit)", frame)
        if cv.waitKey(1) == ord("q"):
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry point for running models.")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image or 'live' to use camera 0 live.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save output image in when not running in 'live' mode",
        default=Path("output_images"),
    )

    sub_parser = parser.add_subparsers(dest="model_type", required=True)

    face_parser = sub_parser.add_parser("face", help="Run face detection.")
    face_parser.add_argument(
        "--model",
        type=Path,
        help="Path to model XML file for cv.CascadeClassifier.",
        default="haarcascade_frontalface_alt.xml",
    )

    pose_parser = sub_parser.add_parser("pose", help="Run pose estimation.")
    pose_parser.add_argument(
        "--detection-quality-threshold",
        type=float,
        default=0.9,
        help="Threshold on goodness of detection for drawing bounding boxes.",
    )
    pose_parser.add_argument(
        "--keypoint-quality-threshold",
        type=float,
        default=0.9,
        help="Threshold on goodness of keypoint for drawing keypoints and skeleton.",
    )

    yolo_parser = sub_parser.add_parser("yolo", help="Run YOLO object detection.")
    yolo_parser.add_argument(
        "--detection-quality-threshold",
        type=float,
        default=0.5,
        help="Threshold on goodness of detection for drawing bounding boxes.",
    )

    segment_parser = sub_parser.add_parser("segment", help="Run semantic segmentation.")
    segment_parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Alpha blending value for blending original image with colorized output image.",
    )

    sunglasses_parser = sub_parser.add_parser("sunglasses", help="Run sunglasses annotator.")
    sunglasses_parser.add_argument(
        "--glasses-image-path",
        type=Path,
        default=Path("cv_models/images/sunglasses.png"),
        help="Path to png file with sunglasses.",
    )
    import sys
    sys.argv = [
        "main.py",                       # Program name placeholder
        "--image", "cv_models/images/solvay.jpg",           # Path to your test image (or "live" for webcam)
        "--output-dir", "output_images", # Optional: output directory
        "face",                          # Model type: face | pose | yolo | segment | sunglasses
        "--model", "cv_models/haarcascade_frontalface_alt.xml"  # Extra model-specific args
    ]


    args = parser.parse_args()

    if args.model_type == "face":
        model = FaceDetector(args.model)
    elif args.model_type == "pose":
        model = PoseEstimator(args.detection_quality_threshold, args.keypoint_quality_threshold)
    elif args.model_type == "yolo":
        model = YOLOObjectDetector(args.detection_quality_threshold)
    elif args.model_type == "segment":
        model = SemanticSegmenter(alpha=args.alpha)
    elif args.model_type == "sunglasses":
        model = SunglassesAnnotator(str(args.glasses_image_path))

    if args.image == "live":
        run_live(model)
    else:
        args.image = Path(args.image)
        if not args.image.exists():
            raise FileNotFoundError(f"Image file not found: {args.image}")
        if not args.output_dir.exists():
            args.output_dir.mkdir(parents=True)
        run_single_image(
            args.image, model, args.output_dir / f"{args.image.stem}_{args.model_type}_out.jpg"
        )
