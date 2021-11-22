import torch
import collections
import numpy as np
import cv2
import os
import pandas as pd
import albumentations as A
import torchvision

from typing import Callable, List
from utils import read_img
from torch.utils.data import Dataset

def get_transforms(augment : bool = False) -> List[Callable]:
    """
    Get the transforms to apply yo the data.

    Args:
        augment (bool): Flag to add data augmentation

    Returns:
        list: list containing the tranformations
    """
    transforms = [
                  A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
                  A.Normalize(
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)
                  ),
    ]

    if augment:
        pass

    return transforms

class WasteImageDataset(Dataset):
    """
    This class defines a dataset for waste object localization

    Args:
        df (pandas.DataFrame): Annotations dataframe
        transforms (List[Callable]): List of transformations
        config (object): Object containing configuration parameters
    """
    def __init__(
        self, 
        df : pd.DataFrame, 
        transforms : List[Callable], 
        config : object
    ) -> None:
        self.df = df
        self.transforms = transforms
        self.config = config

        # Group the annotations by the image identifier
        cols = [col for col in df.columns if col != 'image_id']      
        temp_df = self.df.groupby(['image_id'])[cols]
        # Aggregate in each row all the data corresponding to the same identifier
        self.temp_df = temp_df.agg(lambda x: list(x)).reset_index()
        
        self.image_info = collections.defaultdict(dict)

        for index, row in self.temp_df.iterrows():
            self.image_info[index] = {
                'image_id': np.unique(row['image_id'])[0],
                'height': np.unique(row['height'])[0],
                'width': np.unique(row['width'])[0],
                'area': list(row['area']),
                'iscrowd': list(row['iscrowd']),
                'image_path': os.path.join(self.config.IMGS_PATH, row['filename'][0]),
                'bboxes': list(row['bbox']),
                'polygons': list(row['segmentation']), 
                'categories': list(row['category_id']),
            }

    def _generate_mask(self, height : int, width : int,
                       seg : List[int]) -> np.ndarray:
        """
        Fill the space defined by a list of polygons.

        Args:
            height (int): Image height in pixels.
            width (int): Image width in pixels.
            seg (List[int]): Vertices that define a polygon

        Returns:
            np.ndarray: Mask of the shape represented by the polygons.
        """
        # Object mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Shape of vertices (x_vertices, y_vertices)
        polygon_shape = (int((len(seg) / 2)), 2)
        # Reshape the polygons
        poly = np.array(seg).reshape(polygon_shape).astype(int)
        # Fill the mask
        cv2.fillConvexPoly(mask, poly, 1)

        return mask
    
    def __getitem__(self, idx : int):
        info = self.image_info[idx]

        # Read the image and rotate it if neccesary
        img = read_img(info['image_path'])

        # Check the image is in pytorch format (channels, height, width)
        if img.shape[0] != info['height']:
            raise ValueError('The first dimmension of the image must be the height')
        
        # Create masks
        shape = (len(info['bboxes']), info['height'], info['width'])
        masks = np.zeros(shape, dtype=np.uint8)

        for i, seg in enumerate(info['polygons']):
            # Seg is a list of lists so we must access the first element
            seg = seg[0]
            a_mask = self._generate_mask(info['height'], info['width'], seg)
            a_mask = np.array(a_mask) > 0

            masks[i, :, :] = a_mask

        # Apply the abs to the boxes to avoid having some negative numbers
        boxes = [np.abs(box) for box in info['bboxes']]
        labels = info['categories']

        if self.transforms:
            box_params = A.BboxParams(
                format='coco',
                label_fields=['class_labels'])
            augs = A.Compose(self.transforms, bbox_params=box_params)

            transformed = augs(image=img,
                            masks=list(masks), # This parameter only accepts a list
                            bboxes=boxes,
                            class_labels=labels)

            img = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']

        # Put the channels first, the image is already rotated in format (height, width)
        img = torch.from_numpy(img.transpose(2,0,1)) # channels first

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Change boxes format
        boxes = torchvision.ops.box_convert(boxes, 'xywh', 'xyxy')

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.Tensor(info['image_id']),
            'area': torch.Tensor(info['area']),
            'iscrowd': torch.Tensor(info['iscrowd']),
        }

        return img, target

    def __len__(self):
        return len(self.image_info)