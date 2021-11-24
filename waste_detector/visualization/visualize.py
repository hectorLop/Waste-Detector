#from __future__ import __annotations__
from typing import Tuple
from PIL import Image, ExifTags
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

import matplotlib.pyplot as plt
import colorsys
import numpy as np
import pandas as pd

def get_img(filepath : str, 
            annotations_df : pd.DataFrame, 
            dataset_filepath : str) -> Tuple[Image.Image, int]:
    """
    Read an image and outputs it in the right orientation.

    Args:
        filepath (str): Image filepath inside the dataset.
        annotations_df (pandas.DataFrame): DataFrame containing the annotations
            in COCO format.
        root_imgs_filepath (str): System path where the dataset is stored.

    Returns
        tuple: A tuple containing:
            - Image.Image: Image in PIL format
            - int: image identifier
    """
    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    # Get the image identifier looking in the annotations
    img_id = annotations_df[annotations_df['filename'] == filepath]['image_id'].values[0]
    # Read the image
    img = Image.open(dataset_filepath.IMGS_PATH + filepath)

    # Load and process image metadata
    if img._getexif():
        exif = dict(img._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180,expand=True)
            if exif[orientation] == 6:
                img = img.rotate(270,expand=True)
            if exif[orientation] == 8:
                img = img.rotate(90,expand=True)

    return img, img_id

def plot_img(img : Image.Image,
             annotations_df : pd.DataFrame,
             img_id : int) -> None:
    """
    Plot and image with the object bounding boxes and masks.

    Args:
        img (PIL.Image.Image): Image in PIL format.
        annotations_df (pandas.DataFrame): DataFrame containing the annotations
            in COCO format.
        img_id (int): Image identifier.
    """
    # Slice the annotations dataframe to obtain the annotations that
    # correspond to the actual image
    df = annotations_df[annotations_df['image_id'] == img_id]

    # Show image
    _, ax = plt.subplots(1)
    plt.imshow(img)

    # Loop over the annotations
    for row in df.iterrows():
        # Get the annotation data
        data = row[1]
        # Set a random color for the mask and the bounding box
        color = colorsys.hsv_to_rgb(np.random.random(),1,1)

        # Loop over the polygons
        for seg in data['segmentation']:
            # Group polygons in pairs (x, y)
            poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
            # Patch to fill the polygon
            p = PatchCollection([poly], facecolor=color, edgecolors=color,
                                linewidths=0, alpha=0.4)
            ax.add_collection(p)
            # Patch representing the polygon border
            p = PatchCollection([poly], facecolor='none', edgecolors=color,
                                linewidths=2)
            ax.add_collection(p)

        # Bounding box
        [x, y, w, h] = data['bbox']
        # Add the bounding box to the image
        rect = Rectangle((x ,y), w, h, linewidth=2, edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '--')
        ax.add_patch(rect)

    plt.show()