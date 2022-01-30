from pyexpat import model
import sys
sys.path.insert(0, '../../../icevision/icevision/')

import torch
import icevision.models as models

class Config:
    """
    Config class that defines training parameters and hyperparameters.
    """
    imgs_path = '/home/data/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    img_size = 512
    presize = 512

    # Size of each set (training, test, validation)
    probs = [0.65, 0.2, 0.15]
    seed = 2021

    # Model creation
    model_type = models.ross.efficientdet
    backbone = model_type.backbones.d0
    extra_args = {
        'img_size': img_size
    }
    num_classes = 2

    # Hyperparameters
    batch_size = 8
    epochs = 1
    learning_rate = 0.001
    weight_decay = 0.0000001
    momentum = 0.9