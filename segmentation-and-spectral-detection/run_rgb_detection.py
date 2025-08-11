# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os
import cv2
import numpy as np
import subprocess
from typing import List, Tuple
import numpy.typing as npt

from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.plotting import Colors

from lo.sdk.helpers._import import import_extra
from lo_dataset_reader import DatasetReader
from lo.sdk.api.acquisition.data.formats import LORAWtoLOGRAY12, _debayer
from classifiers.helpers import draw_detections, get_spectra_from_mask


color_palette = Colors()
cuda = import_extra("torch.cuda", extra="yolo")


def get_from_huggingface_model(model_path=None) -> YOLO:
    """
    Get a hugging face model

    Args:
        repo_id: (str) - Name of the repo to download the model from
        filename: (str) - Name of the model file to download

    Returns:
        model: (ultralytics.YOLO) - The loaded model object
    """
    # Initialise face detector through hugging face
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-seg.pt"
    model_path = "yolov8l-seg.pt"

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 85 * 1024 * 1024:
        print(f"Downloading {model_path}...")
        subprocess.run(["wget", yolo_url, "-O", model_path])
    else:
        print(f"{model_path} already exists. Skipping download.")

    device = "cpu"
    if cuda.is_available():
        device = "gpu"
    model = YOLO(model_path)
    model.to(device)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="LO segmentation example",
        epilog="Living Optics 2024",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the root directory of the JSON format dataset.",
    )

    args = parser.parse_args()

    # Initialise LO reader
    reader = DatasetReader(dataset_path=args.dataset_path)

    model = get_from_huggingface_model(model_path="yolov8l-seg.pt")

    for (_, scene_frame, *_), *_ in reader:

        if scene_frame is None:
            break

        scene_frame = np.ascontiguousarray(scene_frame)

        if len(scene_frame.shape) == 3:
            scene_frame = scene_frame.squeeze()
        if np.amax(scene_frame) > 1000:
            scene_frame = LORAWtoLOGRAY12(scene_frame)

        if scene_frame.shape[0] % 2 == 1 or scene_frame.shape[1] % 2 == 1:
            scene_frame = np.dstack([scene_frame, scene_frame, scene_frame])
        else:
            scene_frame = _debayer(scene_frame)

        scene_frame = np.ascontiguousarray(scene_frame).astype(np.uint8)
        scene_frame = cv2.cvtColor(scene_frame, cv2.COLOR_RGB2BGR)

        results = model(scene_frame, imgsz=640, conf=0.4, iou=0.3)
        if results[0].masks is None:
            continue
        segments = results[0].masks.xy
        boxes = results[0].boxes.data.detach().cpu().numpy()
        for (*box, conf, cls_), segment in zip(boxes, segments):
            scene_frame = draw_detections(
                results[0].names, scene_frame, box, conf, cls_, segment
            )

        cv2.imshow("detections", cv2.resize(scene_frame, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)
