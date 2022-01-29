from tests import TRAIN_CLASS, VAL_CLASS
from waste_detector.classifier.train import get_loaders, fit
from waste_detector.classifier.model import CustomEfficientNet
from waste_detector.classifier.config import Config
import pickle
import torch
from torch.utils.data import DataLoader

def test_get_loaders():
    with open(TRAIN_CLASS, "rb") as file:
        train_df = pickle.load(file)

    with open(VAL_CLASS, "rb") as file:
        val_df = pickle.load(file)

    config = Config()
    config.BATCH_SIZE = 1
    config.EPOCHS = 1
    config.DEVICE = 'cpu'

    train_loader, val_loader = get_loaders(train_df, val_df, config)

    assert (train_loader and val_loader)
    assert (isinstance(train_loader, DataLoader) and 
            isinstance(val_loader, DataLoader))

def test_create_model():
    model = CustomEfficientNet("efficientnet_b0", target_size=7,
                               pretrained=False)

    assert model
    assert isinstance(model, torch.nn.Module)

def test_train():
    with open(TRAIN_CLASS, "rb") as file:
        train_df = pickle.load(file)

    with open(VAL_CLASS, "rb") as file:
        val_df = pickle.load(file)

    config = Config()
    config.BATCH_SIZE = 1
    config.EPOCHS = 1
    config.DEVICE = 'cpu'

    train_loader, val_loader = get_loaders(train_df, val_df, config)

    model = CustomEfficientNet("efficientnet_b0", target_size=7,
                               pretrained=False)

    # Clone old parameters
    old_parameters = [param.clone() for param in model.parameters()]

    model, train_loss, val_loss, train_acc, val_acc = fit(model, train_loader, 
                                                          val_loader, config,
                                                          None, None)

    # Check that the model paramters changed
    new_parameters = [param for param in model.parameters()]
    equality = [torch.equal(new_parameters[i], old_parameters[i]) for i in range(len(new_parameters))]

    assert False in equality