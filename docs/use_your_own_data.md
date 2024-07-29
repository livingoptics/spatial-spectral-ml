# Use your own data

You can use the `create_json_dataset` method from `lo.data` to build a dataset that is compatible with our training pipeline.

Simply pass in a list of paths to the `.lo` files you have annotated, along with a list of masks, class names and class numbers.

```python
import cv2
import numpy as np
import os
import os.path as op
from typing import List, Tuple

from lo.data.tools import create_json_dataset
from lo.data.helpers import _debayer
from lo.sdk.api.acquisition.io.open import open as lo_open


# Use this method to extract png images from your .lo data, so you can label it with your favourite data labelling software
def lo_to_png(lo_file_path: str, output_path: str) -> Tuple[List[str], List[str]]:
    """
    Convert an .lo file to a series of pairs of pngs and .lo files and return their file names
    
    Args:
        lo_file_path (str): Path to the .lo file to convert
        output_path (str): Path to a directory to save the extracted data to 
        
    Returns:
        file_paths (List[str]): List of paths to the extracted png frames 
        lo_file_paths (List[str]): List of paths to the extracted lo frames 
    """
    images_dir = op.join(output_path, "images")
    lo_dir = op.join(output_path, "lo_files")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(lo_dir, exist_ok=True)
    
    file_paths = []
    lo_file_paths = []
    with lo_open(lo_file_path, "r", format=".lo") as f:
        for frame_number, frame in enumerate(f):
            info, scene_frame, spectra = frame
            if scene_frame is None:
                break
            
            scene_frame, info = _debayer(scene_frame, info, is_low_res=False)
            scene_frame = scene_frame.astype(np.float32)
            scene_frame = cv2.cvtColor(
                ((scene_frame / np.max(scene_frame)) * 255).astype(np.float32),
                cv2.COLOR_RGB2BGR,
            )
            image_name = f"{op.basename(lo_file_path)}_{frame_number}.png"
            lo_file_name = f"{op.basename(lo_file_path)}_{frame_number}.lo"
            image_frame_path = op.join(images_dir, image_name)
            lo_frame_path = op.join(lo_dir, lo_file_name)
            
            cv2.imwrite(image_frame_path, scene_frame)
            with lo_open(lo_frame_path, "w", format=".lo") as out_f:
                out_f.write(frame)
            file_paths.append(image_frame_path)
            lo_file_paths.append(lo_frame_path)
            
    return file_paths, lo_file_paths
            
     
            
path = "my_new_dataset/train/train.json"

# Create a list of the lo files you have annotated
lo_image_paths = ["image_1.lo", "image_2.lo"]

# Create a list of lists of the mask files for each .lo file
mask_paths = [
    ["image_1_object_1.png", "image_1_object_2.png"],
    ["image_2_object_1.png"]
]

# Create a list of lists of class names, one name per mask
class_names = [
    ["a", "b"], 
    ["c"]
]

# Create a list of lists of class numbers, one name per mask
class_numbers = [
    [0, 1], 
    [2]
]

# Create the JSON dataset
create_json_dataset(path, lo_image_paths, class_names, class_numbers, mask_paths)
```

Masks should be provided as binary segmentation masks, one for each object in the scene, saved in either `png` or `jpg` files.