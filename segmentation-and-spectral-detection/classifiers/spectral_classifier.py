# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os.path as op
import pickle
import time
from typing import Callable, Iterator, List, Union

import matplotlib.pyplot as plt
import numpy as np
from lo.data.helpers import COLOURS
from lo.data.tools import LODataItem
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseSpectralClassifier, _default_label_fn


def plot_labelled_spectra(spectra, labels, class_number_to_label, a):
    """
    Plot labelled spectra with means and standard deviations.

    Args:
        spectra: Spectra data.
        labels: Corresponding labels for spectra.
        class_number_to_label: Mapping from class number to label.
        a: Matplotlib axis to plot on.
    """
    class_spec = {}
    for spec, label in zip(spectra, labels):
        if label not in class_spec:
            class_spec[label] = []
        class_spec[label].append(spec)

    def max_dist(mean, spec_):
        dists = spec_ - mean
        return mean + np.max(dists, axis=0), mean - np.max(-1 * dists, axis=0)

    for label in class_number_to_label.keys():
        if label in class_spec:
            spec = class_spec[label]
            m = np.mean(spec, axis=0)
            s = np.std(spec, axis=0)
            maxi, mini = max_dist(m, spec)
            p = a.plot(
                m,
                label=f"{class_number_to_label[label]}: {label}",
                color=COLOURS[label] / 255,
            )
            a.fill_between(
                np.arange(len(m)), m - s, m + s, color=p[-1].get_color(), alpha=0.7
            )
            a.fill_between(
                np.arange(len(m)), mini, maxi, color=p[-1].get_color(), alpha=0.4
            )
            a.plot(mini, color=p[-1].get_color(), linestyle=":")
            a.plot(maxi, color=p[-1].get_color(), linestyle=":")
    a.legend()


class CPURFClassifier(BaseSpectralClassifier):
    classifier = None
    scaler = None
    reduction_factor = 2

    def load_classifier(self, path):
        """
        Load the classifier and scaler from the given path.

        Args:
            path (str): Path to the classifier and scaler files.
        """
        classifier_path = op.join(path, "classifier.model")
        scaler_path = op.join(path, "scaler.model")
        state_path = op.join(path, "metadata.txt")

        # If the classifier already exists, load it
        if op.exists(classifier_path):
            self.classifier = pickle.load(open(classifier_path, "rb"))
            self.scaler = pickle.load(open(scaler_path, "rb"))
            self.generate_metadata(state_path)

        # If plot_spectra, set up a figure to plot on
        if self.plot_spectra and self.fig is None:
            self.fig, self.ax = plt.subplots(1, 2)
            self.ax[0].set_title("Training Spectra")
            self.ax[1].set_title("Classified Spectra")

    def train(
        self,
        training_data_iterator: Union[Iterator[LODataItem], List[LODataItem]],
        erode_dilate: bool = False,
        iterations: int = 1,
        label_generator_fn: Callable = _default_label_fn,
        include_background_in_training: bool = True,
        n_estimators: int = 50,
        balance_classes: bool = False,
        warm_start: bool = True,
    ) -> None:
        """
        Train a spectral classifier.

        Args:
            training_data_iterator (Iterator[LODataItem]): Iterator with training data.
            erode_dilate (bool): Whether to erode and dilate the mask before indexing to try to remove incorrectly segmented pixels.
            iterations (int): The number of iterations of erosion and dilation to apply - this is not passed to cv2.erode or cv2.dilate. Instead, a loop of erode then dilate is performed "iterations" times.
            label_generator_fn (Callable): Function to generate a class label from an Annotation.
            include_background_in_training (bool): Whether to include the non-labelled spectra in the training data as belonging to the background class (0) - this is necessary for KNN to work well.
            n_estimators (int): The number of trees in the forest.
            balance_classes (bool): Whether to balance the data by class, such that each class has the same amount of training data as the class with the least training samples. Excess samples are discarded.
            warm_start (bool): Whether to warm start the classifier if it has previously been trained. Setting it to False will train a new one from scratch.

        Returns:
            None
        """
        if self.classifier is None or not warm_start:
            self.classifier = OneVsRestClassifier(
                RandomForestClassifier(n_estimators=n_estimators, min_samples_split=2),
                n_jobs=2,
            )
            self.scaler = StandardScaler()
        else:
            self.classifier.estimator.warm_start = True
            print("Warm starting classifier")

        start = time.time()
        print("\tLoading training data")

        all_data, all_labels, masks = self._get_labelled_training_data(
            training_data_iterator,
            erode_dilate,
            iterations,
            label_generator_fn,
            batch_size=None,
            include_background_in_training=include_background_in_training,
        )

        self.class_number_to_label = {
            v: k for k, v in self.label_to_class_number.items()
        }
        all_data_scaled, all_data = self._preprocess(None, all_data)
        all_data_scaled = self.scaler.fit_transform(all_data_scaled)
        print(f"\tData loaded in {time.time() - start:.2f} seconds")

        if balance_classes:
            print("\tBalancing samples per class")
            all_data_scaled, all_labels = self._balance_classes(
                all_data_scaled, all_labels
            )

        num_positive = np.sum(all_labels != 0)
        num_negative = len(all_labels) - num_positive
        print(
            f"\tFitting classifier on {num_positive} positive and {num_negative} negative spectral samples "
        )
        start = time.time()
        self.classifier.fit(all_data_scaled, all_labels)
        print(
            f"\tClassifier trained in {time.time() - start :.2f} seconds - saving artefacts"
        )

        if self.plot_spectra:
            plot_labelled_spectra(
                all_data_scaled, all_labels, self.class_number_to_label, self.ax[0]
            )

        pickle.dump(
            self.classifier,
            open(op.join(self.classifier_path, "classifier.model"), "wb"),
        )
        pickle.dump(
            self.scaler,
            open(op.join(self.classifier_path, "scaler.model"), "wb"),
        )

        np.save(op.join(self.classifier_path, "all_data_scaled.npy"), all_data_scaled)
        np.save(op.join(self.classifier_path, "all_labels.npy"), all_labels)

        self.save_metadata(all_labels, all_data_scaled, self.class_number_to_label)

        preds = self.classifier.predict(all_data_scaled)
        accuracy = np.sum(preds == all_labels) / len(all_labels)
        print(f"Training Accuracy: {accuracy * 100:.2f}%")

    def __call__(
        self, frame, confidence=0.3, sa_factor=4, similarity_filter=True, **kwargs
    ) -> List:
        """
        Call the classifier on a given frame.

        Args:
            frame (Tuple): The input frame.
            confidence (float): Confidence threshold for predictions.
            sa_factor (float): Spectral angle factor.
            similarity_filter (bool): Whether to use similarity filter.
            **kwargs: Additional arguments.

        Returns:
            List: A list of predictions.
        """
        info, scene, o_spectra = frame
        sc = info.sampling_coordinates.astype(np.int32)
        scaled_spectra, spectra = self._preprocess(
            info, o_spectra, do_reflectance=self.do_reflectance
        )
        scaled_spectra = self.scaler.transform(scaled_spectra)
        prob = self.classifier.predict_proba(scaled_spectra)
        labels = np.argmax(prob, axis=-1)
        labels[np.max(prob, axis=-1) < confidence] = 0
        if self.plot_spectra:
            self.ax[1].cla()
            plot_labelled_spectra(
                scaled_spectra, labels, self.class_number_to_label, self.ax[1]
            )
            self.fig.canvas.draw()

        output, class_labels, prob = self._postprocess(
            sc,
            scaled_spectra,
            labels,
            prob,
            sa_factor=sa_factor,
            similarity_filter=similarity_filter,
        )

        return output, class_labels, prob
