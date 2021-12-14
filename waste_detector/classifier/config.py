import torch

class Config:
    IMGS_PATH = '/home/data'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    IMG_SIZE = 512
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0000001
    MOMENTUM = 0.9