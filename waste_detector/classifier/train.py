from typing import Dict
import torch
import numpy as np
import argparse
import gc
import yaml
import pickle
import wandb

from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from waste_detector.classifier.model import CustomEfficientNet
from waste_detector.classifier.dataset import (
    WasteDatasetClassification,
    get_transforms
)
from waste_detector.classifier.config import Config
from waste_detector.classifier.utils import (
    fix_all_seeds
)
from torch.utils.data import DataLoader

def train_step(model, train_loader, config, criterion, optimizer):
    model.train()

    epoch_loss = 0
    y_train = []
    preds = []

        
    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        # Predict
        images = images.to(config.DEVICE)  
        labels = labels.to(config.DEVICE)

        # Set the gradients to zerp before backprop step
        optimizer.zero_grad()

        # Get predictions and calculate the loss
        y_preds = model(images.float())
        y_preds = y_preds.to(config.DEVICE)

        loss = criterion(y_preds, labels)

        # Backward step
        loss.backward()
        optimizer.step()
        
        y_train.append(labels.detach().cpu().numpy())
        preds.append(y_preds.softmax(1).detach().cpu().numpy())

        epoch_loss += loss.item()
        
    y_train = np.concatenate(y_train)
    preds = np.concatenate(preds).argmax(1)
    acc = accuracy_score(y_train, preds)

    return epoch_loss, acc


def val_step(model, val_loader, config, criterion):
    model.eval()

    preds = []
    epoch_loss = 0
    y_test = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 1):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            y_preds = model(images.float())
            y_preds = y_preds.to(config.DEVICE)

            loss = criterion(y_preds, labels)

            y_test.append(labels.cpu().numpy())
            preds.append(y_preds.softmax(1).cpu().numpy())
            epoch_loss += loss.item()
            
    y_test = np.concatenate(y_test)
    preds = np.concatenate(preds).argmax(1)
    acc = accuracy_score(y_test, preds)

    return epoch_loss, acc


def fit(model, train_loader, val_loader, config, filepath, weights):
    model = model.to(config.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.LEARNING_RATE,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    
    if weights is not None:
        weights = torch.FloatTensor(list(weights)).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    n_batches, n_batches_val = len(train_loader), len(val_loader)

    best_loss = np.inf
    val_loss_accum, train_loss_accum = [], []

    with torch.cuda.device(config.DEVICE):
        for epoch in range(1, config.EPOCHS + 1):
            train_loss, train_acc = train_step(model,
                                            train_loader,
                                            config,
                                            criterion,
                                            optimizer)
            
            train_loss = train_loss / n_batches
            train_loss_accum.append(train_loss)
            
            gc.collect()

            val_loss, val_acc = val_step(model,
                                        val_loader,
                                        config,
                                        criterion)
            
            val_loss = val_loss / n_batches_val
            val_loss_accum.append(val_loss)

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            })
            
            gc.collect()

            prefix = f"[Epoch {epoch:2d} / {config.EPOCHS:2d}]"
            print(prefix)
            print(f"{prefix} Train loss: {train_loss:7.5f}. Val loss: {val_loss:7.5f}")
            print(f"{prefix} Train acc: {train_acc:7.5f}. Val acc: {val_acc:7.5f}")

            if val_loss < best_loss:
                best_loss = val_loss
                print(f'{prefix} Save Val loss: {val_loss:7.5f}')
                torch.save(model.state_dict(), filepath)
                
            print(prefix)

    return model, train_loss_accum, val_loss_accum

def get_loaders(df_train, df_val, config=Config):
    ds_train = WasteDatasetClassification(df_train, get_transforms(config, augment=True), config)
    dl_train = DataLoader(ds_train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

    ds_val = WasteDatasetClassification(df_val, get_transforms(config, augment=False), config)
    dl_val = DataLoader(ds_val,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=4)
    return dl_train, dl_val

def train(parameters : Dict):
    fix_all_seeds(4089)
    
    with open(parameters['train_set'], 'rb') as file:
        train_df = pickle.load(file)
        
    with open(parameters['val_set'], 'rb') as file:
        val_df = pickle.load(file)

    train_loader, val_loader = get_loaders(train_df,
                                           val_df)
    
    weights = compute_class_weight('balanced',
                                   classes=np.unique(train_df['category_id'].values),
                                   y=train_df['category_id'].values)

    print('Getting the model')
    model = CustomEfficientNet('efficientnet_b0', target_size=7, pretrained=True)

    wandb.init(
        project='waste_detector',
        entity='hlopez',
        config={
            'learning_rate': Config.LEARNING_RATE,
            'weight_decay': Config.WEIGHT_DECAY,
            'momentum': Config.MOMENTUM,
            'batch_size': Config.BATCH_SIZE,
            'total_epochs': Config.EPOCHS,
            'img_size': Config.IMG_SIZE
        }
    )
    
    print('TRAINING')
    for param in model.parameters():
        param.requires_grad = True
        
    model, train_loss, val_loss = fit(model,
                                      train_loader,
                                      val_loader,
                                      Config,
                                      parameters['checkpoint'],
                                      weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    train(params)