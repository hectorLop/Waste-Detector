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

class Config:
    IMGS_PATH = '/content/drive/MyDrive/Proyectos/Waste-Detector/taco-dataset/'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    BATCH_SIZE=8
    EPOCHS = 30
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.0005

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
            'backbone': get_efficientnetv2_backbone()
        }
    }
}