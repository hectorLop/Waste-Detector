import torch
import albumentations as A
import cv2
import torchvision

from typing import List, Callable
from torch.utils.data import Dataset
from waste_detector.classifier.utils import read_img
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
        augmented_transforms = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(rotate_limit=20),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.RandomBrightnessContrast(),
            A.Blur(blur_limit=(1, 3)),
        ]

    return transforms, augmented_transforms

class WasteDatasetClassification(Dataset):
    def __init__(self, df, transforms, config):
        self.df = df
        self.transforms = transforms
        self.config = config
    
    def __getitem__(self, idx):
        info = self.df.iloc[idx, :]

        # Read the image and rotate it if neccesary
        img = read_img(info['filename'])
        bbox = torch.as_tensor(info['bbox'])
        bbox = torchvision.ops.box_convert(bbox, 'xywh', 'xyxy')
        bbox = torchvision.ops.clip_boxes_to_image(bbox, size=(img.shape[:2]))
        img = crop_img_to_bbox(img, bbox)
        label = info['category_id']
        
        if self.transforms:
            common_trans, augmented_trans = self.transforms
            common_trans = A.Compose(common_trans)
            # augs = A.Compose(self.transforms)
            
            transformed = common_trans(image=img)
            img = transformed['image']

            if augmented_trans and label != 0:
                augmented_trans = A.Compose(augmented_trans)
                transformed = augmented_trans(image=img)
                img = transformed['image']

        # Put the channels first, the image is already rotated in format (height, width)
        img = torch.from_numpy(img.transpose(2,0,1)) # channels first

        label = torch.as_tensor(label, dtype=torch.int64)

        return img, label

    def __len__(self):
        return len(self.df)