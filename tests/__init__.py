import os
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder

ANNOTATIONS = _TEST_ROOT + '/data/test_annotations.json'
IMG_DIR = _TEST_ROOT + '/data/'
INDICES = _TEST_ROOT + '/data/indices.json'

TRAIN_CLASS = _TEST_ROOT + '/data/train_class.pkl'
VAL_CLASS = _TEST_ROOT + '/data/val_class.pkl'