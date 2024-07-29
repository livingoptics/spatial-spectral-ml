
# Spatial Spectral Machine Learning

Showcases how machine and deep learning application can be built on top of the Living Optics Camera.

### All examples are built on top of our huggingface dataset and model repository [🤗 Living Optics](https://huggingface.co/LivingOptics)


<center>

| RGB segmentation with spectral detection                                           |
|:---------------------------------------------------------------------------------------:|
| [![segmentation-spectral-detection](./segmentation-and-spectral-detection/media/fruit-recoginition-spectra-apples.gif)](./segmentation-and-spectral-detection/run_spectral_detection.py)  |


| RGB segmentation                                          |
|:---------------------------------------------------------:|
| [![segmentation-rgb](./segmentation-and-spectral-detection/media/fruit-recognition-apples.gif)](./segmentation-and-spectral-detection/run_rgb_detection.py) |

</center>

This example illustrates the utilisation of spatial and spectral information from the Living Optics Camera for a detection and segmentation task. By integrating a simple spectral 
classifier, trained with minimal data, into a semantic segmentation pipeline, this method achieves subclass recognition of objects. This level of detail is unattainable with 
standard RGB data alone.

## Getting started
- Please read the [Getting started guide](https://developer.livingoptics.com/getting-started/)
- Register to download the SDK and sample data [here](https://www.livingoptics.com/register-for-download-sdk/)
- Registered as a user? Read the [documentation](https://docs.livingoptics.com/)
- Have a look at our other application examples [here](github)



## Install
These examples require specific model dependencies. Install them using the following commands.

#### Requres Python version >=3.8 and <= 3.11

Download the SDK [here](https://cloud.livingoptics.com/shared-resources?file=software/lo_sdk-1.4.0-dist.tgz).

Make a request for the lo-data package through Huggingface [here](https://huggingface.co/spaces/LivingOptics/README/discussions/3).

Follow the installation instructions provided [here](https://docs.livingoptics.com/sdk/install-guide.html#custom-python-environment) to install the python SDK.

Additional install packages required:

```bash
source venv/bin/activate
pip install 'install/lo_sdk-{version}-py3-none-any.whl[yolo]'
pip install lo_data-{version}-py3-none-any.whl
pip install -r segmentation-and-spectral-detection/requirements.txt
```

To enable GPU support see the [installation guide from Pytorch](https://pytorch.org/get-started/locally/)

## Usage

Add the segmentation-and-spectral-detection package to your python path 
```bash
 export PYTHONPATH="${PYTHONPATH}:segmentation-and-spectral-detection"
```

### Download Training Dataset from Hugging Face

To download the training dataset, use the following code:

```python
from huggingface_hub import hf_hub_download
dataset_path = hf_hub_download(repo_id="LivingOptics/hyperspectral-fruit", filename="train", repo_type="dataset")
print(dataset_path)
```

OR 

```bash
mkdir -p hyperspectral-fruit
huggingface-cli download LivingOptics/hyperspectral-fruit include 'train/*' --repo-type dataset --local-dir hyperspectral-fruit
```

Run the following command if on orin:

```bash
mkdir -p /datastore/lo/share/samples/hyperspectral-fruit
huggingface-cli download LivingOptics/hyperspectral-fruit --repo-type dataset --local-dir /datastore/lo/share/samples/hyperspectral-fruit
```

#### For details on the dataset navigate to  [🤗 Living Optics hyperspectral fruit](https://huggingface.co/datasets/LivingOptics/hyperspectral-fruit)

### Download demo data:

To download all demo videos for fruit detection run the following.

```python
from huggingface_hub import hf_hub_download
demo_data_path = hf_hub_download(repo_id="LivingOptics/hyperspectral-fruit", filename='demo-videos', repo_type="dataset")
print(demo_data_path)
```

OR 

```bash
mkdir -p hyperspectral-fruit
huggingface-cli download LivingOptics/hyperspectral-fruit include 'demo-videos/*' --repo-type dataset --local-dir hyperspectral-fruit
```

To download a specfic demo video files navigate to [🤗 Living Opitcs demo dataset](https://huggingface.co/LivingOptics/hyperspectral-fruit/demo-videos)

### Fit a Classifier

This step trains a spectral classifier on the LO dataset.

```bash
python ./spatial-and-spectral-detection/fit_classifier.py --model_path ./fruit-classifier --dataset_path ./hyperspectral-fruit
```
### Run the Model

To run the model, execute:

```bash
python ./segmentation-and-spectral-detection/run_spectral_detection.py --source ./hyperspectral-fruit/demo-videos/apple-varieties-tomatoes.lo --model_path ./fruit-classifier --plot_classifier_raw_outputs
```
This inference example demonstrates a simple integration between the YOLO SAM model and a spectral classifier to achieve subclass classification and improved recognition while 
preserving the semantic understanding of the model. 

To stream data from the Living Optics Camera into the classifier, you must pass a calibration to the script as shown below.

```bash
python ./segmentation-and-spectral-detection/run_spectral_detection.py --model_path ./fruit-classifier --calibration /datastore/lo/share/calibrations/latest_calibration 
```

It is advised when steaming to run this script with a calibration frame using the '--calibration_file_path' arg. [See here for more details about field calibrations](https://docs.livingoptics.com/product/tech-notes/decode-calibration/decode-calibration.html?h=calibration)

### Notebook end to end example:

This notebook will take you through the end-to-end process of fitting and running spectrally enhanced segmentation.

```bash
jupyter notebook ./segmentation-and-spectral-detection/fit_and_run_detectors.ipynb
```

### Coming soon:
- A workflow for training on your own LO data.
- Orin AGX GPU support.
- Support for the spectrally enhanced segmentation in the LO analysis tool, to enable real time applications.


## Extensions

### Build a Custom Classifier

Below is a template to build your own classifier, to test and explore other classification algorithms such as SVMs, AdaBoost, Neural Networks .... 

```python
from .classifiers.base import BaseSpectralClassifier

def default_label_fn(ann):
    return ann.class_name


class Classifier(BaseSpectralClassifier):

    classifier = None
    scaler = None
    reduction_factor = 3
    classes = None

    def load_classifier(self, path):
        # If the classifier already exists, load it
        if op.exists(path):
            ##############################################
            # ADD MODE LOAD CODE, example self.classifier = pickle.load(open(classifier_path, "rb"))
            ##############################################
            self.generate_metadata(state_path)      

    def train(
        self,
        training_data_iterator: Iterator[LODataItem],
        erode_dilate: bool = False,
        iterations: int = 1,
        label_generator_fn: Callable = default_label_fn,
        include_background_in_training: bool = True,
        **kwargs,
    ) -> None:
        """
        Train a spectral classifier
        Args:
            training_data_iterator (Iterator[LODataItem]): iterator with training data
            erode_dilate (bool): whether to erode and dilate the mask before indexing to try to remove incorrectly segmented pixels
            iterations (int): the number of iterations of erosion and dilation to apply - this is not passed to cv2.erode or cv2.dilate. Instead, a loop of erode then dilate is
             performed "iterations" times.
            label_generator_fn (Callable): Function to generate a class label from an Annotation
            include_background_in_training (bool): whether to include the non-labelled spectra in the training data as belonging to the background class (0)
        Returns:
            None
        """

        if self.classifier is None:
            ####################################################
            # Init classifier object here
            # ADD CUSTOM INIT FOR CLASSIFIER, EXAMPLE: self.classifier = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=2) 
            ####################################################
           
        # retrieve data
        all_data, all_labels, masks = self._get_labelled_training_data(
            training_data_iterator,
            erode_dilate,
            iterations,
            label_generator_fn,
            batch_size=None,
            include_background_in_training=include_background_in_training,
        )

        all_data_scaled, _ = self._preprocess(None, all_data)
        
        class_number_to_label = {v: k for k, v in self.label_to_class_number.items()}
        
        ####################################################
        #  ADD CUSTOM MODEL TRAINING CODE 
        ####################################################

        # Save trained model
        
        # ADD MODEL SAVE CODE HERE EXAMPLES: pickle.dump(self.classifier,open(op.join(self.classifier_path, "classifier.model"), "wb"))

        # Save mean spectra and max distance to the mean

        self.save_metadata(all_labels, all_data_scaled, class_number_to_label)
        
        self.load_classifier(self.classifier_path)

    def __call__(
        self, frame, confidence=0.3, sa_factor=10, similarity_filter=True,  **kwargs
    ) -> List:
        info, scene, o_spectra = frame
        sc = info.sampling_coordinates.astype(np.int32)
        scaled_spectra, spectra = self._preprocess(info, o_spectra, do_reflectance=self.do_reflectance)
        #scaled_spectra = spectra


        ####################################################
        #  ADD CUSTOM MODEL INFERENCE CODE 
        ####################################################

        # EXAMPLE: 
        # prob = self.classifier.predict_proba(scaled_spectra)
        # labels = np.argmax(prob,axis=-1)  
        # labels[np.max(prob,axis=-1) < confidence] = 0

        output, class_labels, prob = self._postprocess(sc, spectra, labels, prob, sa_factor=sa_factor, similarity_filter=similarity_filter)
        
        return output, class_labels, prob


```

### Use Your Own Data
See the [use your own data readme](./docs/use_your_own_data.md)

## Resources

- [Developer documentation](https://developer.livingoptics.com/)
- [Developer resources](https://www.livingoptics.com/developer)
- [Product documentation](https://docs.livingoptics.com/) for registered users.

## Contribution Guidelines
We welcome contributions to enhance this project. Please follow these steps to contribute:

**Fork the Repository**: Create a fork of this repository to your GitHub account.

**Create a Branch**: Create a new branch for your changes.

**Make Changes**: Make your changes and commit them with a clear and descriptive message.

**Create a Pull Request**: Submit a pull request with a description of your changes.

## Support

For any questions, contact us at [Living Optics support](https://www.livingoptics.com/support).