import sys
sys.path.insert(0, '../../../icevision/icevision/')

import torch
import icevision.models as models

class Config:
    """
    Config class that defines training parameters and hyperparameters.
    """
    IMGS_PATH = '/home/data/'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    IMG_SIZE = 512
    PRESIZE = 512

    # Size of each set (training, test, validation)
    PROBS = [0.65, 0.2, 0.15]
    SEED = 2021

    # Model creation
    MODEL_TYPE = models.ross.efficientdet
    BACKBONE = MODEL_TYPE.backbones.d0
    EXTRA_ARGS = {
        'img_size': IMG_SIZE
    }
    NUM_CLASSES = 2

    # Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 1
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0000001
    MOMENTUM = 0.9