import pytest

import icevision
from icevision.tfms.albumentations import Adapter
from icevision.data.record_collection import RecordCollection
from waste_detector.config import BASE_DIR
from waste_detector.object_detection.config import Config
from waste_detector.object_detection.utils import get_transforms, get_splits

def test_get_transforms():
    """
    Test the get_transforms() function.
    """
    train_tfms, val_tfms, test_tfms = get_transforms(Config)

    # Assert transforms are not None
    assert (train_tfms and val_tfms and test_tfms)
    # Assert transforms are albumentations Adapters
    assert (isinstance(train_tfms, Adapter) and isinstance(val_tfms, Adapter)
            and isinstance(test_tfms, Adapter))
    # Assert transforms list are not empty
    assert (train_tfms.tfms_list and val_tfms.tfms_list and test_tfms.tfms_list)

def test_get_splits(mocker):
    """
    Test the get_splits() function.
    """
    train, test, val = get_splits(annotations=str(BASE_DIR) + '/TACO-master/data/annotations.json',
                                  img_dir=str(BASE_DIR) + '/TACO-master/data/')

    assert (train and test and val)
    assert(isinstance(train, RecordCollection) and 
           isinstance(test, RecordCollection) and 
           isinstance(val, RecordCollection))