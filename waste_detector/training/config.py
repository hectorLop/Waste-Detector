import torch

class Config:
    IMGS_PATH = '/content/drive/MyDrive/Proyectos/Waste-Detector/taco-dataset/'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    BATCH_SIZE=8
    EPOCHS = 30
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.0005