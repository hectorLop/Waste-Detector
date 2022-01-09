import os
import random

import numpy as np
import PIL
import torch
from PIL import ExifTags, Image


def crop_img_to_bbox(image, bbox):
    img = image.copy()
    bbox = np.array(bbox).astype(int)
    cropped_img = PIL.Image.fromarray(img).crop(bbox)

    return np.array(cropped_img)


def fix_all_seeds(seed: int = 4496) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def read_img(filepath: str):
    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            break

    image = Image.open(filepath)

    # Load and process image metadata
    if image._getexif():
        exif = dict(image._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            if exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            if exif[orientation] == 8:
                image = image.rotate(90, expand=True)

    return np.array(image)
