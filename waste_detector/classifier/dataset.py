import torch
import numpy as np
import collections
import albumentations as A
import cv2
import torchvision

from typing import List, Callable
from torch.utils.data import Dataset
from waste_detector.training.utils import read_img
from waste_detector.classifier.utils import crop_img_to_bbox

class CustomNormalization(A.ImageOnlyTransform):
    def _norm(self, img):
        return img / 255.

    def apply(self, img, **params):
        return self._norm(img)

def get_transforms(config, augment : bool = False) -> List[Callable]:
    """
    Get the transforms to apply yo the data.

    Args:
        augment (bool): Flag to add data augmentation

    Returns:
        List[Callable]: list containing the tranformations
    """
    transforms = [
        A.Resize(config.IMG_SIZE, config.IMG_SIZE,
                interpolation=cv2.INTER_NEAREST),
        CustomNormalization(p=1),
    ]

    if augment:
        pass

    return transforms

class WasteDatasetClassification(Dataset):
    def __init__(self, df, transforms, config):
        self.df = df
        self.transforms = transforms
        self.config = config

        #cols = [col for col in df.columns if col != 'image_id']
        #self.temp_df = self.df.groupby(['image_id'])[cols].agg(lambda x: list(x)).reset_index()
        
        #self.image_info = collections.defaultdict(dict)

        #for index, row in self.temp_df.iterrows():
         #   self.image_info[index] = {
         #       'image_id': np.unique(row['image_id'])[0],
          #      'height': np.unique(row['height'])[0],
           #     'width': np.unique(row['width'])[0],
            #    'area': list(row['area']),
             #   'iscrowd': list(row['iscrowd']),
              #  'image_path': row['filename'][0],
         #       'bboxes': list(row['bbox']),
          #      'categories': list(row['category_id']),
           # }
    
    def __getitem__(self, idx):
        info = self.df.iloc[idx, :]
        #info = self.image_info[idx]

        # Read the image and rotate it if neccesary
        img = read_img(info['filename'])
        bbox = torch.as_tensor(info['bbox'])
        bbox = torchvision.ops.box_convert(bbox, 'xywh', 'xyxy')
        bbox = torchvision.ops.clip_boxes_to_image(bbox, size=(img.shape[:2]))
        img = crop_img_to_bbox(img, bbox)
        label = info['category_id']
        
        if self.transforms:
                augs = A.Compose(self.transforms)
                transformed = augs(image=img)

                img = transformed['image']

        # Put the channels first, the image is already rotated in format (height, width)
        img = torch.from_numpy(img.transpose(2,0,1)) # channels first

        label = torch.as_tensor(label, dtype=torch.int64)

        return img, label

    def __len__(self):
        return len(self.df)