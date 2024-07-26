# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import logging
import os
from pathlib import Path

import numpy as np
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import _debayer
from lo.sdk.api.acquisition.io.open import open as loopen
from lo.sdk.api.camera.camera import LOCamera

try:
    import cupy

    xp = cupy
    logging.warning(
        "Cupy is successfully imported. "
        "GPU will be used during the execution of this program."
    )
except ImportError:
    logging.warning(
        "GPU enabled cupy failed to import, falling back to CPU. "
        "The execution may be slow."
    )
    import numpy

    xp = numpy


def percentile_norm(im, low=1, high=95):
    """
    Normalize the image based on percentile values.

    Args:
        im (np.ndarray): The input image.
        low (int): The lower percentile for normalization.
        high (int): The higher percentile for normalization.

    Returns:
        np.ndarray: The normalized image.
    """
    im[..., 0] = im[..., 0] - np.percentile(im[::100, ::10, 0], low)
    im[..., 0] = im[..., 0] / np.percentile(im[::100, ::10, 0], high)
    im[..., 1] = im[..., 1] - np.percentile(im[::100, ::10, 1], low)
    im[..., 1] = im[..., 1] / np.percentile(im[::100, ::10, 1], high)
    im[..., 2] = im[..., 2] - np.percentile(im[::100, ::10, 2], low)
    im[..., 2] = im[..., 2] / np.percentile(im[::100, ::10, 2], high)
    return np.clip(im, 0, 1) * 255


class LOReader:
    def __init__(
        self,
        calib_folder=None,
        source=None,
        calibration_frame=None,
        frame_rate=20,
        exposure=50,
        gain=0,
    ):
        """
        Initialize the LOReader.

        Args:
            calib_folder (str): Path to the calibration folder.
            source (str): Source file or camera device.
            calibration_frame (str, optional): Path to the calibration frame.
            frame_rate (int, optional): Frame rate for the camera. Defaults to 20.
            exposure (int, optional): Exposure time for the camera. Defaults to 50.
            gain (int, optional): Gain for the camera. Defaults to 0.
        """
        self.decoder = None
        if calib_folder is not None:
            calibration_folder = Path(calib_folder).as_posix()
            self.decoder = SpectralDecoder.from_calibration(
                calibration_folder, calibration_frame
            )
        self.file_handler = None
        self.lo_format = False
        if source is not None:
            self.file_handler = loopen(source, "r")
            if os.path.splitext(source)[1] == ".lo":
                self.lo_format = True

        self.source = LOCamera(file=self.file_handler)

        # Open camera and turn on stream
        self.source.open()
        self.source.stream_on()

        if source is None:
            self.source.frame_rate = frame_rate * 1000 * 1000
            self.source.gain = gain
            self.source.exposure = exposure * 1000

    def __del__(self):
        """
        Destructor to ensure the stream is turned off and the source is closed.
        """
        self.source.stream_off()
        self.source.close()

        if self.file_handler is not None:
            self.file_handler.close()
            self.file_handler = None

    def __len__(self):
        """
        Return the number of frames in the file if available.

        Returns:
            int: Number of frames, or -1 if not available.
        """
        if self.file_handler is not None:
            return len(self.file_handler)
        return -1

    def get_next_frame(self, return_raw=False):
        """
        Get the next frame from the source.

        Args:
            return_raw (bool, optional): Whether to return the raw frame data. Defaults to False.

        Returns:
            tuple: Information, scene frame, and spectra.
        """
        try:
            if self.file_handler is None:
                frame = self.source.get_frame()
            else:
                frame = self.file_handler.read()

            if self.lo_format:
                info, scene_frame, spectra = frame
            else:
                info, scene_frame, spectra = self.decoder(frame)

            if return_raw:
                return info, scene_frame, spectra

            debayered = _debayer(scene_frame[:, :, 0], info)
            debayered = percentile_norm(debayered.astype(np.float32))
            scene_frame = xp.asarray(debayered).astype(np.uint8)
            return info, scene_frame, spectra

        except Exception as e:
            logging.warning(f"Received exception {e}")
            return None, None, None

    def seek_frame(self, frame_idx):
        """
        Seek to a specific frame index in the file.

        Args:
            frame_idx (int): Frame index to seek to.
        """
        if self.file_handler is None:
            logging.warning("Cannot seek frame index when running camera live.")
        else:
            self.file_handler.seek(frame_idx)

    def get_wavelengths(self):
        """
        Get the wavelengths from the calibration data.

        Returns:
            np.ndarray: Array of wavelengths.
        """
        return self.decoder.calibration.wavelengths

    def get_sampling_coordinates(self):
        """
        Get the sampling coordinates from the calibration data.

        Returns:
            np.ndarray: Array of sampling coordinates.
        """
        return self.decoder.sampling_coordinates
