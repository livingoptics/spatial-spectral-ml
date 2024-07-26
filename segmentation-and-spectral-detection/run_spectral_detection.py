# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import cv2
import numpy as np
from classifiers.fastsam.model import FastSAM
from classifiers.helpers import draw_detections, get_spectra_from_mask
from classifiers.spectral_classifier import CPURFClassifier as classifier

if __name__ == "__main__":
    import argparse

    from reader import LOReader

    parser = argparse.ArgumentParser(
        prog="LO spectral classifier",
        epilog="Living Optics 2024",
    )
    parser.add_argument(
        "--source",
        default=None,
        type=str,
        help="Path to an .lo or .loraw file to test the trained classifier with, default is from streaming camera",
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
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration folder. Only used if a test source is provided.",
    )

    parser.add_argument(
        "--calibration_file_path",
        default=None,
        type=str,
        help="Path to field calibration frame Only used if a test source is provided.",
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

    args = parser.parse_args()

    # Initialise LO reader
    reader = LOReader(
        args.calibration, args.source, calibration_frame=args.calibration_file_path
    )

    # Set camera parameters - only relevant when streaming.
    reader.source.frame_rate = int(10000e3)  # μhz
    reader.source.exposure = int(100000e3)  # μs
    reader.source.gain = 100

    # Load trained spectral classifier
    classifier = classifier(
        classifier_path=args.model_path,
        plot_spectra=False,
        do_reflectance=True,
    )

    # load segment anything model developed by ultralytics.
    model = FastSAM("FastSAM-x.pt")

    while True:

        info, scene_frame, spectra = reader.get_next_frame()

        if scene_frame is None:
            break

        scene_frame = np.ascontiguousarray(scene_frame)

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

        cv2.imshow("segmentation result", scene_frame)
        cv2.waitKey(1)
