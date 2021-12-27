from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import random
import os
import torch

from waste_detector.object_detection.config import Config
from icevision.parsers.coco_parser import COCOBBoxParser
from icevision.data.data_splitter import RandomSplitter
from icevision.data.record_collection import RecordCollection
import icevision.tfms as tfms

def fix_all_seeds(seed : int = 4496) -> None:
    """
    Fix the seeds to make the results reproducible.

    Args:
        seed (int): Desired seed number. Default is 4496.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_splits(annotations : str, img_dir : str,
               config : Config = Config) -> Tuple[RecordCollection]:
    """
    Split the data given the annotations in COCO format.

    Args:
        annotations (str): Annotations filepath.
        img_dir (str): Images filepath.
        config (Config): Config object

    Returns:
        Tuple[RecordCollection]: Tuple containing:
            - (RecordCollection): Training record
            - (RecordCollection): Testing record
            - (RecordCollection): Validation record
    """
    parser = COCOBBoxParser(annotations_filepath=annotations,
                            img_dir=img_dir)
    splitter = RandomSplitter(probs=config.PROBS,
                              seed=config.SEED)

    train, test, val = parser.parse(data_splitter=splitter,
                                    autofix=True)

    return train, test, val

def get_transforms(config : Config) -> Tuple[tfms.A.Adapter]:
    """
    Get the transformations for each set.

    Args:
        config (Config): Configuration object

    Returns:
        Tuple[tfms.A.Adapter]: Tuple containing:
            - (tfms.A.Adapter): Training transforms
            - (tfms.A.Adapter): Validation transforms
            - (tfms.A.Adapter): Testing transforms
    """
    train_tfms = tfms.A.Adapter([
        *tfms.A.aug_tfms(size=config.IMG_SIZE, presize=config.PRESIZE),
        tfms.A.Normalize()
    ])

    valid_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(config.IMG_SIZE),
        tfms.A.Normalize()
    ])

    test_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(config.IMG_SIZE),
        tfms.A.Normalize()
    ])

    return train_tfms, valid_tfms, test_tfms