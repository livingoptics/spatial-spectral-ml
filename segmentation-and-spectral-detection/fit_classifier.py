# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os
import numpy as np
import cv2

from lo.sdk.analysis.ml.models.spectral_classifier import CPURFClassifier as classifier
from lo.sdk.api.acquisition.data.formats import LORAWtoLOGRAY12, _debayer
from lo_dataset_reader import DatasetReader, rle_to_mask


def label_fn(ann):
    """
    Generate a class label string from the metadata of an annotation
    Args:
        ann (lo.data.tools.tools.Annotation):

    Returns:
        label (str): the class label for the annotation
    """
    return ann["category_name"]

def percentile_norm(im: np.ndarray, low: int = 1, high: int = 99) -> np.ndarray:
    """
    Normalise the image based on percentile values.

    Args:
        im (xp.ndarray): The input image.
        low (int): The lower percentile for normalization.
        high (int): The higher percentile for normalization.

    Returns:
        xp.ndarray: The normalised image.
    """

    ## ::30 is subsampling
    low, high = np.percentile(im[::40, ::40], (low, high), axis=(0, 1))
    im = (im - low) / (high - low)
    return np.clip(im, 0, 1) * 255

def visualise(scene, pred):
    """
    Args:
        scene: Original image (H, W) or (H, W, 3)
        pred: Dictionary mapping class IDs to coordinate arrays
    Returns:
        vis_img: RGB image with overlaid predictions
    """
    # Create empty mask
    h, w = scene.shape[:2]
    
    # Define color map (adjust according to your actual class labels)
    color_map = {
        0: (255, 255, 0),       # background (cyan)
        1: (0, 255, 0),     # class 1 (green) - grapes
        2: (255, 0, 0),     # class 2 (blue) - tray
        3: (0, 0, 255),     # class 3 (red) - tyvec
    }
    
    # Prepare the scene image
    if len(scene.shape) == 3:
        scene = scene.squeeze()
    if np.amax(scene) > 1000:
        # .loraw scene frame
        scene = LORAWtoLOGRAY12(scene)

    if scene.shape[0] % 2 == 1 or scene.shape[1] % 2 == 1:
        output = np.dstack([scene, scene, scene])
    else:
        output = _debayer(scene)

    output = percentile_norm(output.astype(np.float32)).astype(np.uint8)
    output = output[:, :, [2, 1, 0]]
    scene = np.ascontiguousarray(output)
    
    # Create a copy of the scene to draw on
    vis_img = scene.copy()
    
    # Draw each class with its corresponding color
    for class_id, coords in pred.items():            
        # Ensure coordinates are integers and within bounds
        coords = coords.astype(int)
        valid_coords = (coords[:, 0] < h) & (coords[:, 1] < w) & (coords[:, 0] >= 0) & (coords[:, 1] >= 0)
        coords = coords[valid_coords]
        
        # Draw each point as a small circle
        for y, x in coords:
            cv2.circle(vis_img, (x, y), radius=5, color=color_map[class_id], thickness=-1)
    
    # Add legend
    legend_height = 60
    legend = np.zeros((legend_height, scene.shape[1], 3), dtype=np.uint8)

    class_names = {
        0: "Background",
        1: "Grapes",
        2: "Tray",
        3: "Tyvec"
    }

    font_scale = 0.9
    font_thickness = 2
    box_width = 25
    box_height = 25
    y_box_top = 17
    y_text = y_box_top + box_height - 5

    x_pos = 10
    spacing = 16
    for class_id, color in color_map.items():
        x_box = x_pos
        x_text = x_box + box_width + spacing

        # Draw rectangle
        cv2.rectangle(legend, (x_box, y_box_top), (x_box + box_width, y_box_top + box_height), color, -1)

        # Draw outlined text
        text = class_names[class_id]
        cv2.putText(legend, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness + 1, lineType=cv2.LINE_AA)
        cv2.putText(legend, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

        x_pos += box_width + spacing + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][0] + 40

    # Combine image and legend
    vis_img = np.vstack([vis_img, legend])

    return vis_img


if __name__ == "__main__":
    import argparse
    import time

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
        "--dataset_path",
        type=str,
        help="Path to the root directory of the JSON format dataset.",
    )
    parser.add_argument(
        "--source",
        default="",
        type=str,
        help="Path to an .lo or .loraw file to test the trained classifier with",
    )
    args = parser.parse_args()

    # Load dataset using the custom DatasetReader class
    reader = DatasetReader(dataset_path=args.dataset_path)

    all_spectra = []
    all_labels = []

    class_number_to_label = {
        0: "background",
        1: "grapes",
        2: "tray",
        3: "tyvec",

    }
    label_to_class_number = {v: k for k, v in class_number_to_label.items()}

    for (info, scene_frame, spectra, *_), _, annotations, *_ in reader:
        h, w = scene_frame.shape[:2]
        coordinates = info.sampling_coordinates.astype(int)  # shape: (N, 2)

        # Create a label mask for the whole frame
        label_mask = np.zeros((h, w), dtype=np.uint8)

        for ann in annotations:
            class_name = label_fn(ann)
            if class_name not in label_to_class_number:
                continue

            label_index = label_to_class_number[class_name]

            if ann.get("segmentation"):
                mask = rle_to_mask(ann["segmentation"], (h, w))
                label_mask[(mask > 0) & (label_mask == 0)] = label_index

        # Map each spectrum point to its label
        for i, (y, x) in enumerate(coordinates):
            if 0 <= y < h and 0 <= x < w:
                label = label_mask[y, x]
                if label in class_number_to_label:
                    spectrum = spectra[i]  # shape: (96,)
                    all_spectra.append(spectrum)
                    all_labels.append(label)

    # Convert to arrays
    all_spectra = np.stack(all_spectra)  # (N, 96)
    all_labels = np.array(all_labels)    # (N,)

    # Instantiate Classifier
    classifier = classifier(
        classifier_path=args.model_path,
        plot_spectra=False,
        do_reflectance=False,
        class_number_to_label=class_number_to_label,
    )

    print("Starting training")
    classifier.train(
        all_spectra=all_spectra,
        all_labels=all_labels,
        n_estimators=70,
        warm_start=args.warm_start,
    )

    # Print class labels
    for k, v in classifier.metadata.items():
        print(k, v[0])

    # Iterate over frames and display the classifications
    start = time.time()
    count = 0
    for (info, scene_frame, spectra, *_), *_ in reader:

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
        vis = visualise(scene_frame, pred[0])

        cv2.imshow("display", cv2.resize(vis, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

        save_dir = "visualisations"
        save_path = os.path.join(save_dir, f"output_{count:03d}.jpg")
        cv2.imwrite(save_path, vis)

        end = time.time()
        print(f"FPS: {1 / (end - start):.2f}")

        count += 1
        start = end
