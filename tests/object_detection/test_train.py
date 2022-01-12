import pytest

from waste_detector.object_detection.train import get_data_loaders
from waste_detector.config import BASE_DIR
from waste_detector.object_detection.config import Config

from torch.utils.data import DataLoader

def test_get_data_loaders():
    annotations_file = str(BASE_DIR) + '/TACO-master/data/annotations.json'
    img_dir = str(BASE_DIR) + '/TACO-master/data/'

    train_dl, valid_dl, test_dl = get_data_loaders(annotations=annotations_file,
                                                   img_dir=img_dir,
                                                   config=Config)

    assert (train_dl and valid_dl and test_dl)
    assert (isinstance(train_dl, DataLoader) and 
            isinstance(valid_dl, DataLoader) and
            isinstance(test_dl, DataLoader))