# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os
import cv2
import subprocess
import numpy as np

from classifiers.fastsam.model import FastSAM
from classifiers.helpers import draw_detections, get_spectra_from_mask

from lo.sdk.analysis.ml.models.spectral_classifier import CPURFClassifier as classifier
from lo_dataset_reader import DatasetReader
from lo.sdk.api.acquisition.data.formats import LORAWtoLOGRAY12, _debayer



if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(
        prog="LO spectral classifier",
        epilog="Living Optics 2024",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to save the trained model artefact to.",
    )
    parser.add_argument(
        "--plot_classifier_raw_outputs",
        action="store_true",
        help="plot raw specta outputs",
    )
    parser.add_argument(
        "--threshold",
        default=0.4,
        type=float,
        help="Confidence threshold for the spectral classifier.",
    )
    parser.add_argument(
        "--sa_factor",
        default=4,
        type=float,
        help="Spectral angle multiplier below which to consider a spectrum as a true classification for a particular class",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the root directory of the JSON format dataset.",
    )

    args = parser.parse_args()

    # Initialise LO reader
    reader = DatasetReader(dataset_path=args.dataset_path)

    class_label_to_number = {
        0: "background_class",
    }

    # Load trained spectral classifier
    classifier = classifier(
        classifier_path=args.model_path,
        plot_spectra=False,
        do_reflectance=True,
        class_number_to_label=class_label_to_number,
    )

    # load segment anything model developed by ultralytics.
    fastsam_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt"
    filename = "FastSAM-x.pt"

    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        subprocess.run(["wget", fastsam_url, "-O", filename])
    else:
        print(f"{filename} already exists. Skipping download.")
    model = FastSAM("FastSAM-x.pt")
    
    for (info, scene_frame, spectra, *_), *_ in reader:

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

        results = model(
            scene_frame, device="cpu", retina_masks=True, imgsz=480, conf=0.6, iou=0.9
        )
        masks = results[0].masks.data.detach().cpu().numpy()
        segs = results[0].masks.xy
        boxes = results[0].boxes.data.detach().cpu().numpy()

        # Inference with the spectral classifier
        metadata_out, classes, probs = classifier(
            (info, scene_frame, spectra),
            confidence=args.threshold,
            sa_factor=args.sa_factor,
        )

        # Visualise spectral classifier raw results
        if args.plot_classifier_raw_outputs:
            classifier_output, _ = classifier.visualise(
                (info, scene_frame, spectra), metadata_out
            )
            cv2.imshow("spectral classifier result", classifier_output)

        # Assign classes to objects identified by Segment Anything
        for bbox, mask, seg in zip(boxes, masks, segs):
            bb_probs = get_spectra_from_mask(probs, info.sampling_coordinates, mask)
            bb_class = np.argmax(np.median(bb_probs, 0))
            if bb_class == 0:
                continue
            bb_prob = bb_probs[:, bb_class].mean(0)
            if bb_prob < args.threshold:
                continue
            scene_frame = draw_detections(
                classifier.classes, scene_frame, bbox, bb_prob, bb_class, seg
            )

        cv2.imshow("segmentation result", cv2.resize(scene_frame, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)
