import torch

from waste_detector.training.models import (
    create_efficientdet_model,
    create_custom_faster_rcnn,
    get_efficientnetv2_backbone
)
from waste_detector.training.dataset import (
    efficientdet_collate_fn,
    faster_rcnn_collate_fn
)

import icevision.models as models

class Config:
    IMGS_PATH = '/home/data/'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    IMG_SIZE = 512
    PRESIZE = 512

    PROBS = [0.65, 0.2, 0.15]
    SEED = 2021

    MODEL_TYPE = models.ross.efficientdet
    BACKBONE = MODEL_TYPE.backbones.d0
    EXTRA_ARGS = {
        'img_size': IMG_SIZE
    }
    NUM_CLASSES = 2

    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.0000001
    MOMENTUM = 0.9

MODELS_FUNCTIONS = {
    'efficientdet': {
        'collate_fn': efficientdet_collate_fn,
        'model_fn': create_efficientdet_model,
        'params': {
            'image_size': 512,
            'architecture': 'efficientdet_d1'
        }
    },
    'faster_rcnn': {
        'collate_fn': faster_rcnn_collate_fn,
        'model_fn': create_custom_faster_rcnn,
        'params': {
            'backbone': get_efficientnetv2_backbone
        }
    }
}