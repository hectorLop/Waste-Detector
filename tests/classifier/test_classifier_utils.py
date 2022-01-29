import pytest
import json
import pandas as pd
import numpy as np
import torch
import torchvision

from waste_detector.classifier.utils import read_img, crop_img_to_bbox

from tests import ANNOTATIONS, IMG_DIR

@pytest.fixture
def get_img_and_bbox():
    with open(ANNOTATIONS, 'r') as file:
        data = json.load(file)

    annotations_df = pd.DataFrame(data['annotations'])
    images_df = pd.DataFrame(data['images'])

    annot = annotations_df.iloc[0, :]
    img_path = images_df[images_df['id'] == annot['image_id']]['file_name']
    img_path = IMG_DIR + img_path

    img_bbox = annot['bbox']

    return img_path.values[0], img_bbox

def test_read_img(get_img_and_bbox):
    img_path, _ = get_img_and_bbox
    img = read_img(img_path)

    assert img is not None
    assert isinstance(img, np.ndarray)

def test_crop_img_to_bbox(get_img_and_bbox):
    img_path, bbox = get_img_and_bbox

    # Get the original image
    orig_img = read_img(img_path)

    # Transform the bbox 
    bbox = torch.as_tensor(bbox)
    bbox = torchvision.ops.box_convert(bbox, "xywh", "xyxy")
    bbox = torchvision.ops.clip_boxes_to_image(bbox, size=(orig_img.shape[:2]))
    
    img = crop_img_to_bbox(orig_img, bbox)

    assert img is not None
    assert isinstance(img, np.ndarray)
    # Assert that the cropped image size is smaller than the original
    assert orig_img.shape[0] > img.shape[0]
    assert orig_img.shape[1] > img.shape[1]