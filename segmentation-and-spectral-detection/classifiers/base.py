# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os.path as op
import pickle
from typing import Callable, Dict, Iterator, List, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from lo.data.helpers import COLOURS, rle2mask
from lo.data.tools import Annotation, LODataItem
from lo.sdk.api.acquisition.data.coordinates import SceneToSpectralIndex
from lo.sdk.api.acquisition.io.open import open as lo_open

from .helpers import spectral_angle_nd_to_vec


def _default_label_fn(ann: Annotation):
    """
    Default label function to return the category of an annotation.

    Args:
        ann (Annotation): An annotation object.

    Returns:
        str: The category of the annotation.
    """
    return ann.category


def init_metadata(path):
    """
    Initialize metadata from a given path.

    Args:
        path (str): The path to the metadata file.

    Returns:
        Tuple[Dict, Union[None, np.ndarray]]: The metadata dictionary and reference array if present.
    """
    if op.exists(path):
        metadata = pickle.load(open(path, "rb"))
    reference = None
    if "reference" in list(metadata.keys()):
        reference = np.array(metadata.pop("reference"))
    return metadata, reference


class BaseSpectralClassifier:
    metadata = {}
    include_background_in_training = False

    def __init__(
        self,
        classifier_path,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        thickness=3,
        bg_colour=(1, 1, 1),
        plot_spectra: bool = False,
        do_reflectance: bool = True,
    ):
        """
        Initialize the BaseSpectralClassifier.

        Args:
            classifier_path (str): Path to the classifier.
            font (int): Font type for visualizations.
            font_scale (float): Font scale for visualizations.
            thickness (int): Thickness of the text in visualizations.
            bg_colour (tuple): Background color for visualizations.
            plot_spectra (bool): Whether to plot spectra.
            do_reflectance (bool): Whether to perform reflectance.
        """
        self.default_illuminant = None
        self.do_reflectance = do_reflectance
        self.fig = None
        self.plot_spectra = plot_spectra
        self.classifier_path = classifier_path
        self.class_number_to_label = {0: "background_class", 99999: "reference"}
        self.label_to_class_number = {
            v: k for k, v in self.class_number_to_label.items()
        }
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.bg_colour = bg_colour
        self.reference = None
        self.text_width = 60
        self.text_height = 50
        self.load_classifier(classifier_path)

    def set_default_illuminant(self, ill):
        """
        Set the default illuminant.

        Args:
            ill: Illuminant value to set.
        """
        self.default_illuminant = ill

    def load_classifier(self, path):
        """
        Load the classifier from a given path.

        Args:
            path (str): Path to the classifier file.
        """
        pass

    def _preprocess(self, info, data: npt.NDArray[np.float32], do_reflectance: bool = False):
        """
        Preprocess the data.

        Args:
            info: Information about the data.
            data: Data to preprocess.
            do_reflectance (bool): Whether to perform reflectance.

        Returns:
            Tuple: Scaled data and original data.
        """
        if do_reflectance:
            data = data / self.get_illuminant(info, data, None, [])

        data = data.reshape(
            -1, data.shape[1] // self.reduction_factor, self.reduction_factor
        ).mean(-1)
        data_scaled = data / np.sum(data, axis=-1, keepdims=True)

        return data_scaled, data

    def _postprocess(
        self,
        sc,
        spectra: npt.NDArray[np.float32],
        labels: list,
        probs: npt.NDArray[np.float32],
        sa_factor=1.0,
        similarity_filter=True,
    ):
        """
        Postprocess the data.

        Args:
            sc: Spectral coefficients.
            spectra: Spectra data.
            labels: Labels for the spectra.
            probs: Probabilities.
            sa_factor (float): Spectral angle factor.
            similarity_filter (bool): Whether to use similarity filter.

        Returns:
            Tuple: Output dictionary, class labels, and probabilities.
        """
        output = {}
        class_labels = np.zeros(spectra.shape[0])
        for k, v in self.metadata.items():
            index = np.arange(spectra.shape[0])
            mask = np.ones([spectra.shape[0]], dtype=bool)
            if v is None:
                continue
            if similarity_filter:
                thr = v[2] * sa_factor
                seg_mask = spectral_angle_nd_to_vec(spectra[labels == k], v[1]) < thr
                index = index[labels == k][seg_mask]
                locs = sc[labels == k][seg_mask]
            else:
                index = index[labels == k]
                locs = sc[labels == k]
            if len(locs) == 0:
                continue
            if not isinstance(locs, np.ndarray):
                locs = locs.get()

            output[k] = locs
            mask[index] = 0
            class_labels[index] = k
        return output, class_labels, probs

    def save_metadata(
        self,
        all_labels: npt.NDArray[np.int32],
        all_data_scaled: npt.NDArray[np.float32],
        class_number_to_label: dict,
    ):
        """
        Save metadata to a file.

        Args:
            all_labels: All labels.
            all_data_scaled: All scaled data.
            class_number_to_label: Dictionary mapping class numbers to labels.
        """
        if self.include_background_in_training:
            cls_ids = np.unique(all_labels)[1:]  # Ignore the background
        else:
            cls_ids = np.unique(all_labels)

        cls_means = [all_data_scaled[all_labels == idx].mean(0) for idx in cls_ids]

        cls_dist = [
            np.std(
                spectral_angle_nd_to_vec(
                    all_data_scaled[all_labels == idx], cls_means[i]
                )
            )
            for i, idx in enumerate(cls_ids)
        ]
        cls_res = {idx: [cls_means[i], cls_dist[i]] for i, idx in enumerate(cls_ids)}

        self.metadata = {
            k: [v, cls_res[k][0], cls_res[k][1]]
            for k, v in class_number_to_label.items()
            if k in cls_res
        }

        pickle.dump(
            self.metadata, open(op.join(self.classifier_path, "metadata.txt"), "wb")
        )

    def _get_bg_fg_spectra_and_labels(
        self,
        info,
        spectra: npt.NDArray[np.float32],
        masks: List[npt.NDArray[Union[bool, int]]],
        class_labels: List[str],
    ) -> Tuple[npt.NDArray, List[npt.NDArray], npt.NDArray, List[npt.NDArray]]:
        """
        Extract the foreground and background spectra and generate the class labels associated with each spectrum.

        Args:
            info: Metadata of an LO frame.
            spectra (npt.NDArray[float]): List of spectra.
            masks (List[npt.NDArray[Union[bool, int]]]): A list of segmentation masks.
            class_labels (List[str]): A list of class labels - one for each mask.

        Returns:
            bg_spectra (npt.NDArray): Background spectra [N, 96].
            fg_spectra (List[npt.NDArray]): Foreground spectra [P, Q*, 96] - Q will vary per object in the scene depending on its size.
            bg_labels (npt.NDArray): Background labels [N, 96].
            fg_labels (List[npt.NDArray]): Foreground labels [P, Q*] - Q will vary per object in the scene depending on its size.
        """
        sc = info.sampling_coordinates.astype(np.int32)

        sampling = np.zeros_like(masks[0])
        sampling[sc[:, 0], sc[:, 1]] = 1

        spectra_labels = [
            np.where(
                (mask & sampling)[sc[:, 0], sc[:, 1]], self.label_to_class_number[l], 0
            ).reshape(len(sc))
            for l, mask in zip(class_labels, masks)
        ]

        dilated_spectra_labels = [
            np.where(
                (cv2.dilate(mask, kernel=np.ones((5, 5)), iterations=3) & sampling)[
                    sc[:, 0], sc[:, 1]
                ],
                self.label_to_class_number[l],
                0,
            ).reshape(len(sc))
            for l, mask in zip(class_labels, masks)
        ]

        bg_indices = np.invert(np.any(dilated_spectra_labels, axis=0))
        bg_spectra = spectra[bg_indices]
        bg_labels = np.zeros(len(bg_spectra), dtype=np.int32)

        fg_spectra = [spectra[i != 0] for i in spectra_labels if np.any(i)]
        fg_labels = [
            np.zeros(len(s), dtype=np.int32) + self.label_to_class_number[l]
            for s, l in zip(fg_spectra, class_labels)
        ]

        if len(np.shape(bg_spectra)) == 3:
            bg_spectra = np.squeeze(bg_spectra)

        return bg_spectra, fg_spectra, bg_labels, fg_labels

    def _iterate_anns(
        self,
        lo_data_item,
        class_number: int,
        scene_frame: npt.NDArray[np.float32],
        label_generator_fn: Callable,
        iterations: int,
        erode_dilate: bool,
    ) -> Tuple[List[npt.NDArray], List[str], int]:
        """
        Iterate over the annotations of an LODataItem, creating a list of masks and class labels and keeping count of the number of classes seen.

        Args:
            lo_data_item (LODataItem): The data item to iterate the annotations of.
            class_number (int): The one more than the maximum class number already seen.
            scene_frame (npt.NDArray): The scene frame of lo_data_item.
            label_generator_fn (Callable): A function to convert annotation metadata into a class label.
            iterations (int): The number of iterations of erosion and dilation to apply.
            erode_dilate (bool): Whether to erode and then dilate the labelled masks before extracting spectra to try to reduce the number of incorrectly labelled pixels being used.

        Returns:
            masks (List[npt.NDArray]): A list of class segmentation masks aligned to the scene_frame.
            class_labels (List[str]): A list of class names, the same length as masks (i.e. one class name per mask).
            class_number (int): One more than the highest class number already seen.
        """
        shape = np.shape(scene_frame)
        shape = (shape[0], shape[1])

        masks = []
        class_labels = []
        ann: Annotation
        for idx, ann in enumerate(lo_data_item.annotations):
            mask = rle2mask(ann.segmentation, shape=shape)
            label = label_generator_fn(ann)
            if label not in self.label_to_class_number:
                self.label_to_class_number[label] = ann.class_number

            if erode_dilate:
                for iteration in range(iterations):
                    mask = cv2.erode(mask, np.ones((5, 5)), iterations=1)
                    mask = cv2.dilate(mask, np.ones((5, 5)), iterations=1)
            masks.append(mask)
            class_labels.append(label)
        return masks, class_labels, class_number

    @classmethod
    def _concatenate_data(
        clas,
        all_data: Union[None, npt.NDArray[np.float32]],
        all_labels: Union[None, npt.NDArray[np.int32]],
        bg_spectra: npt.NDArray[np.float32],
        fg_spectra: List[npt.NDArray[np.float32]],
        bg_labels: npt.NDArray[np.int32],
        fg_labels: List[npt.NDArray[np.int32]],
        include_background_in_training: bool,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """
        Combine the input data into a single set of data and labels.

        Args:
            all_data (Union[None, npt.NDArray[float]]): The spectra to train on [N, 96].
            all_labels (Union[None, npt.NDArray[int]]): The class labels of each spectrum [N].
            bg_spectra (npt.NDArray[float]): Background spectra [N, 96].
            fg_spectra (List[npt.NDArray[float]]): Foreground spectra [P, Q*, 96] - Q will vary per object in the scene depending on its size.
            bg_labels (npt.NDArray[int]): Background labels [N, 96].
            fg_labels (List[npt.NDArray[int]]): Foreground labels [P, Q*] - Q will vary per object in the scene depending on its size.
            include_background_in_training (bool): Whether to include the background spectra and labels as part of the returned data.

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[int]]: All data and labels combined.
        """
        if all_data is None:
            if include_background_in_training:
                all_data = np.concatenate((bg_spectra, *fg_spectra), axis=0)
                all_labels = np.concatenate((bg_labels, *fg_labels), axis=0)
            else:
                all_data = np.concatenate([*fg_spectra], axis=0)
                all_labels = np.concatenate([*fg_labels], axis=0)
        else:
            if include_background_in_training:
                all_data = np.concatenate((all_data, bg_spectra, *fg_spectra), axis=0)
                all_labels = np.concatenate((all_labels, bg_labels, *fg_labels), axis=0)
            else:
                all_data = np.concatenate((all_data, *fg_spectra), axis=0)
                all_labels = np.concatenate((all_labels, *fg_labels), axis=0)

        return all_data, all_labels

    def _get_labelled_training_data(
        self,
        training_data_iterator: Iterator[LODataItem],
        erode_dilate: bool = False,
        iterations: int = 1,
        label_generator_fn: Callable = _default_label_fn,
        batch_size: Union[None, int] = None,
        include_background_in_training: bool = True,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], List[npt.NDArray[bool]]]:
        """
        Get a set of training spectra and their associated class labels.

        Args:
            training_data_iterator (Iterator[LODataItem]): The training data to iterate over.
            erode_dilate (bool): Whether to erode and then dilate the labelled masks before extracting spectra to try to reduce the number of incorrectly labelled pixels being used.
            iterations (int): The number of iterations of erosion and dilation to apply.
            label_generator_fn (Callable): A function to convert annotation metadata into a class label.
            batch_size (Union[None, int]): The size of batches to return - The default None returns all training data in a single batch.
            include_background_in_training (bool): Whether to include the non-labelled spectra in the training data as belonging to the background class (0) - this is necessary for KNN to work well but other classifiers may not want a background class.

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[int], List[npt.NDArray[bool]]]: All data, all labels, and masks for each annotation.
        """
        all_data = None
        all_labels = None
        class_number = (
            np.max(
                [
                    k
                    for k in self.class_number_to_label.keys()
                    if k != self.label_to_class_number["reference"]
                ]
            )
            + 1
        )
        lo_data_item: LODataItem
        for i, lo_data_item in enumerate(training_data_iterator):
            with lo_open(op.join(lo_data_item.root_dir, lo_data_item.lo_url), "r") as f:
                info, scene_frame, spectra = f.read()

                masks, class_labels, class_number = self._iterate_anns(
                    lo_data_item,
                    class_number,
                    scene_frame,
                    label_generator_fn,
                    iterations,
                    erode_dilate,
                )

                if self.do_reflectance is not None:
                    spectra = spectra / self.get_illuminant(
                        info, spectra, masks, class_labels
                    )

                bg_spectra, fg_spectra, bg_labels, fg_labels = (
                    self._get_bg_fg_spectra_and_labels(
                        info,
                        spectra,
                        masks,
                        class_labels,
                    )
                )
                if fg_spectra:
                    all_data, all_labels = self._concatenate_data(
                        all_data,
                        all_labels,
                        bg_spectra,
                        fg_spectra,
                        bg_labels,
                        fg_labels,
                        include_background_in_training,
                    )

                    if batch_size is not None and (i + 1) % batch_size == 0:

                        if "reference" in self.label_to_class_number:
                            all_labels[
                                all_labels == self.label_to_class_number["reference"]
                            ] = self.label_to_class_number["background_class"]
                        all_labels[
                            all_labels > self.label_to_class_number["reference"]
                        ] -= 1
                        for k, v in self.label_to_class_number.items():
                            if v > self.label_to_class_number["reference"]:
                                self.label_to_class_number[k] -= 1

                        self.class_number_to_label = {
                            v: k
                            for k, v in self.label_to_class_number.items()
                            if k != "reference"
                        }

                        return all_data, all_labels, masks

        if batch_size is None:
            if "reference" in self.label_to_class_number:
                all_labels[all_labels == self.label_to_class_number["reference"]] = (
                    self.label_to_class_number["background_class"]
                )
            all_labels[all_labels > self.label_to_class_number["reference"]] -= 1
            for k, v in self.label_to_class_number.items():
                if v > self.label_to_class_number["reference"]:
                    self.label_to_class_number[k] -= 1

            self.class_number_to_label = {
                v: k for k, v in self.label_to_class_number.items() if k != "reference"
            }

            return all_data, all_labels, masks

    def visualise(
        self,
        frame: tuple,
        pred: dict,
    ) -> npt.NDArray:
        """
        Generic inference method that calls a classifier on an LO frame and then plots the point wise classifications.

        Args:
            frame (Tuple): .lo decoded frame.
            pred (Dict): 1D array of predictions for the spectra in the scene.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: Annotated scene frame and mask view.
        """
        info, scene_frame, spectra = frame
        if not isinstance(scene_frame, np.ndarray):
            scene_frame = scene_frame.get()
        frame_vis = scene_frame.copy()
        mask_view = np.zeros_like(scene_frame)
        mask_view = cv2.rectangle(
            mask_view, (10, 10), (self.text_width, self.text_height), self.bg_colour, -1
        )
        for k in self.metadata.keys():
            if self.metadata[k][0] not in ["background_class", "reference"]:
                v = pred.get(k)
                c = COLOURS[k]
                c = (int(c[0]), int(c[1]), int(c[2]))
                # Draw legend
                mask_view = cv2.rectangle(
                    mask_view, (20, 20 + 40 * (k - 1)), (40, 40 + 40 * (k - 1)), c, -1
                )
                mask_view = cv2.putText(
                    mask_view,
                    self.metadata[k][0],
                    (50, 40 + 40 * (k - 1)),
                    self.font,
                    self.font_scale,
                    c,
                    self.thickness,
                )
                if v is None:
                    continue

                for pts in v[:, ::-1]:
                    cv2.circle(mask_view, tuple(pts), 3, c, 2)

        frame_vis = frame_vis.mean(-1)
        frame_vis[mask_view.sum(-1) != 0] = 0
        frame_vis = np.repeat(frame_vis[:, :, np.newaxis], 3, axis=-1).astype(np.uint8)

        frame_vis = frame_vis + mask_view

        return frame_vis, mask_view

    def get_illuminant(
        self,
        info,
        spectra: npt.NDArray[np.float32],
        masks: npt.NDArray[np.int32],
        class_labels: list,
    ):
        """
        Calculate the illuminant based on reference class or default illuminant.

        Args:
            info: Metadata of the LO frame.
            spectra: Spectra data.
            masks: array of segmentation masks.
            class_labels: List of class labels.

        Returns:
            np.ndarray: Calculated illuminant.
        """
        if "reference" in class_labels:
            sc = info.sampling_coordinates.astype(np.int32)
            indexer = SceneToSpectralIndex(sc)

            sampling = np.zeros_like(masks[0])
            sampling[sc[:, 0], sc[:, 1]] = 1
            mask = np.squeeze(
                np.asarray(masks)[
                    np.argwhere(np.asarray(class_labels) == "reference")[0]
                ]
            )
            indexes = indexer(np.argwhere((mask & sampling)))
            illuminant = np.mean(spectra[indexes], axis=0)

        elif self.default_illuminant is not None:
            illuminant = self.default_illuminant

        else:
            sumi = np.sum(spectra, axis=-1)
            illuminant = np.mean(
                spectra[np.argwhere(sumi > np.percentile(sumi, 95))], axis=0
            )

        return illuminant

    def _balance_classes(
        self, all_data: npt.NDArray[np.float32], all_labels: npt.NDArray[np.int32]
    ):
        """
        Balance classes by subsampling the dataset to have the same number of samples for each class.

        Args:
            all_data: All spectra data.
            all_labels: All labels.

        Returns:
            Tuple: Balanced data and labels.
        """
        class_counts = []
        class_names = []
        for class_name, class_idx in self.label_to_class_number.items():
            class_counts.append(np.sum(all_labels == class_idx))
            class_names.append(class_name)
            print(f"\tClass: {class_name} - {class_counts[-1]} total samples")

        smallest_idx = (
            np.argmin(
                np.array(class_counts)[
                    np.argwhere(
                        np.logical_and(
                            np.asarray(class_names) != "background_class",
                            np.asarray(class_names) != "reference",
                        )
                    )
                ]
            )
            + 2
        )
        if class_names[smallest_idx] == "background_class":
            return all_data, all_labels
        else:
            all_sample_idxs = []
            for class_name in class_names:

                if (
                    np.sum(all_labels == self.label_to_class_number[class_name])
                    >= class_counts[smallest_idx]
                ):
                    sample_idxs = self._sub_sample_dataset(
                        all_labels,
                        class_counts[smallest_idx],
                        self.label_to_class_number[class_name],
                    )
                else:
                    sample_idxs = np.squeeze(
                        np.argwhere(
                            all_labels == self.label_to_class_number[class_name]
                        )
                    )

                if np.any(sample_idxs):
                    all_sample_idxs += sample_idxs
        print(
            f"\tAfter balancing, all classes have {class_counts[smallest_idx]} samples"
        )
        return all_data[all_sample_idxs], all_labels[all_sample_idxs]

    def _sub_sample_dataset(
        self, all_labels: npt.NDArray[np.float32], num_samples: int, class_idx: int
    ):
        """
        Sub-sample the dataset to have a fixed number of samples for a given class.

        Args:
            all_labels: All labels.
            num_samples (int): Number of samples to extract.
            class_idx (int): Index of the class to subsample.

        Returns:
            List: Indices of the subsampled dataset.
        """
        indexes = np.squeeze(np.argwhere(all_labels == class_idx))
        return list(np.random.choice(indexes, num_samples, replace=False))

    def _predict(self, frame, **kwargs) -> Dict[int, npt.NDArray]:
        """
        Abstract method to be implemented for making predictions on a given frame.

        Args:
            frame: The input frame.
            **kwargs: Additional arguments.

        Returns:
            Dict[int, npt.NDArray]: Predictions for each class.
        """
        pass

    def __call__(self, frame, **kwargs) -> List:
        """
        Call the classifier on a given frame.

        Args:
            frame: The input frame.
            **kwargs: Additional arguments.

        Returns:
            List: A list of predictions.
        """
        pass

    def generate_metadata(self, state_path: str):
        """
        Generate metadata from a given state path.

        Args:
            state_path (str): The path to the state file.
        """
        self.metadata, self.reference = init_metadata(state_path)

        self.class_number_to_label = {0: "background_class", 99999: "reference"}
        self.class_number_to_label.update({k: v[0] for k, v in self.metadata.items()})
        self.label_to_class_number = {
            v: k for k, v in self.class_number_to_label.items()
        }
        self.text_width = (
            max(
                [
                    cv2.getTextSize(v[0], self.font, self.font_scale, self.thickness)[
                        0
                    ][0]
                    for k, v in self.metadata.items()
                    if k != "background_class" and k != "reference"
                ]
            )
            + 60
        )

        self.text_height = (
            40
            * len(
                [
                    k
                    for k in self.metadata
                    if k != "background_class" and k != "reference"
                ]
            )
            + 10
        )

        self.classes = [v[0] for v in self.metadata.values()]
