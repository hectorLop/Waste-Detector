import numpy as np
import random
import os
import torch

from PIL import Image, ExifTags

def fix_all_seeds(seed : int = 4496) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def read_img(filepath : str):
    #Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    I = Image.open(filepath)

    # Load and process image metadata
    if I._getexif():
        exif = dict(I._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                I = I.rotate(180,expand=True)
            if exif[orientation] == 6:
                I = I.rotate(270,expand=True)
            if exif[orientation] == 8:
                I = I.rotate(90,expand=True)

    return np.array(I)

def annotations_to_device(annotations, device):
    if isinstance(annotations, list):
        new_annot = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    elif isinstance(annotations, dict):
        new_annot = {}
        
        for k, v in annotations.items():
            values = [val.to(device) for val in v]
            new_annot[k] = values
            
    return new_annot