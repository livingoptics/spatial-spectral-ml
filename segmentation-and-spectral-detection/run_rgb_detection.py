# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import pickle
import time

import cv2
import numpy as np
from classifiers.helpers import draw_detections, get_spectra_from_mask
from classifiers.spectral_classifier import CPURFClassifier as classifier
from lo.sdk.helpers._import import import_extra
from reader import LOReader
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.plotting import Colors

color_palette = Colors()

huggingface_hub = import_extra("huggingface_hub", extra="yolo")
hf_hub_download = huggingface_hub.hf_hub_download
ultralytics = import_extra("ultralytics", extra="yolo")
YOLO = ultralytics.YOLO
cuda = import_extra("torch.cuda", extra="yolo")


def get_from_huggingface_model(
    repo_id: str = "jaredthejelly/yolov8s-face-detection",
    filename: str = "YOLOv8-face-detection.pt",
    model_path=None,
) -> YOLO:
    """
    Get a hugging face model

    Args:
        repo_id: (str) - Name of the repo to download the model from
        filename: (str) - Name of the model file to download

    Returns:
        model: (ultralytics.YOLO) - The loaded model object
    """
    # Initialise face detector through hugging face
    if model_path is None:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
    device = "cpu"
    if cuda.is_available():
        device = "gpu"
    model = YOLO(model_path)
    model.to(device)
    return model


if __name__ == "__main__":
    import argparse

    from reader import LOReader

    parser = argparse.ArgumentParser(
        prog="LO segmentation example",
        epilog="Living Optics 2024",
    )
    parser.add_argument(
        "--source",
        default=None,
        type=str,
        help="Path to an .lo or .loraw file to test the trained classifier with",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration folder. Only used if a test source is provided.",
    )

    parser.add_argument(
        "--calibration_file_path",
        default="",
        type=str,
        help="Path to field calibration frame Only used if a test source is provided.",
    )

    args = parser.parse_args()

    # Initialise LO reader
    reader = LOReader(
        args.calibration, args.source, calibration_frame=args.calibration_file_path
    )

    # Set camera parameters - only relevant when streaming.
    reader.source.frame_rate = int(10000e3)  # μhz
    reader.source.exposure = int(100000e3)  # μs
    reader.source.gain = 100

    model = get_from_huggingface_model(model_path="yolov8l-seg.pt")

    while True:

        info, scene_frame, spectra = reader.get_next_frame()

        if scene_frame is None:
            break

        scene_frame = np.ascontiguousarray(scene_frame)

        results = model(scene_frame, imgsz=640, conf=0.4, iou=0.3)
        if results[0].masks is None:
            continue
        segments = results[0].masks.xy
        boxes = results[0].boxes.data.detach().cpu().numpy()
        for (*box, conf, cls_), segment in zip(boxes, segments):
            scene_frame = draw_detections(
                results[0].names, scene_frame, box, conf, cls_, segment
            )

        cv2.imshow("detections", scene_frame)

        cv2.waitKey(1)
