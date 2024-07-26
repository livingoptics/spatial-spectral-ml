# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os

import cv2
from classifiers.spectral_classifier import CPURFClassifier as classifier
from lo.data.tools import LOJSONDataset


def label_fn(ann):
    """
    Generate a class label string from the metadata of an annotation
    Args:
        ann (lo.data.tools.tools.Annotation):

    Returns:
        label (str): the class label for the annotation
    """
    return ann.class_name


if __name__ == "__main__":
    import argparse
    import time

    from reader import LOReader

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
        "--warm_start",
        action="store_true",
        help="Whether to run a warm start when training the model - requires that a pretrained model exist at --model_path, otherwise training will be done from scratch",
    )
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Whether to balance the data used to train the random forest, such that each class has the same number of training samples as the class with the least number of "
        "samples.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the root directory of the JSON format dataset.",
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
    parser.add_argument(
        "--source",
        default="",
        type=str,
        help="Path to an .lo or .loraw file to test the trained classifier with",
    )
    args = parser.parse_args()

    # Get dataset:
    dataset = LOJSONDataset(args.dataset_path)
    training_data = dataset.load("train")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Instantiate Classifier
    classifier = classifier(
        classifier_path=args.model_path,
        plot_spectra=False,
        do_reflectance=True,
    )

    print("Starting training")
    # Train on data
    classifier.train(
        training_data,
        erode_dilate=True,
        iterations=1,
        label_generator_fn=label_fn,
        include_background_in_training=True,
        n_estimators=70,
        balance_classes=args.balance_classes,
        warm_start=args.warm_start,
    )

    # Print class labels
    for k, v in classifier.metadata.items():
        print(k, v[0])

    if args.source:
        if not args.calibration_file_path:
            args.calibration_file_path = None
        # Set up the reader
        reader = LOReader(
            args.calibration,
            args.source,
            calibration_frame=args.calibration_file_path,
        )

        # Iterate over frames and display the classifications
        start = time.time()
        count = 0
        while True:
            info, scene_frame, spectra = reader.get_next_frame()

            if scene_frame is None:
                break
            frame = (info, scene_frame, spectra)
            start = time.time()
            pred = classifier(
                frame,
                confidence=0.7,
                sa_factor=4,
                similarity_filter=False,
            )
            end = time.time()
            print(f"call FPS: {1 / (end - start):.2f}")
            vis = classifier.visualise(frame, pred[0])

            cv2.imshow("display", vis[0])
            cv2.waitKey(1)

            end = time.time()
            print(f"FPS: {1 / (end - start):.2f}")

            count += 1
            start = end
