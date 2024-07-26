# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import cv2
import numpy as np
from lo.sdk.helpers.import_numpy_or_cupy import xp
from ultralytics.utils.plotting import Colors

color_palette = Colors()


def get_spectra_from_mask(values, sampling_coordinates, mask):
    sampling_coordinates = np.int32(sampling_coordinates)
    in_mask = mask[sampling_coordinates[:, 0], sampling_coordinates[:, 1]]
    return values[in_mask.astype(bool), :]


def masks2segments(masks):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
        masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

    Returns:
        segments (List): list of segment masks.
    """
    segments = []
    x = masks
    c = cv2.findContours(x.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
        0
    ]  # CHAIN_APPROX_SIMPLE
    if c:
        c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
    else:
        c = np.zeros((0, 2))  # no segments found
    return c.astype(np.float32)


def draw_detections(classes, img, box, score, class_id, mask):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        classes (List[str]): List of class names.
        img (numpy.ndarray): The input image to draw detections on.
        box (np.ndarray): Detected bounding box.
        score (float): Corresponding detection score.
        class_id (int): Class ID for the detected object.

    Returns:
        None
    """

    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

    # Retrieve the color for the class ID
    color = color_palette(class_id, bgr=True)
    im_canvas = img.copy()
    cv2.polylines(img, np.int32([mask]), True, (0, 0, 0), 2)  # white borderline
    cv2.fillPoly(im_canvas, np.int32([mask]), color)

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Create the label text with class name and score
    label = f"{classes[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Calculate the position of the label text
    label_x = int(x1)
    label_y = int(y1 - 10 if y1 - 10 > label_height else y1 + 10)

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + label_height),
        color,
        cv2.FILLED,
    )

    # Draw the label text on the image
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    img = cv2.addWeighted(im_canvas, 0.3, img, 0.7, 0)

    return img


def spectral_angle_nd_to_vec(spectral_list, reference_spectrum):
    """Calculates the spectral angle difference in radians between each row
        of the spectral list and the reference spectrum.

    Args:
        spectral_list (np.ndarray): shape (N_spectra, N_channels)
        reference_spectrum (np.ndarray): shape (N_channels)

    Returns:
        list of SAM scores (np.ndarray): in radians shape (N_spectra)
    """

    if isinstance(spectral_list, np.ndarray):
        xp = np

    return xp.arccos(
        xp.clip(
            xp.dot(spectral_list, reference_spectrum)
            / xp.linalg.norm(spectral_list, axis=1)
            / xp.linalg.norm(reference_spectrum),
            0,
            1,
        )
    )
