from typing import Dict
import torch
import numpy as np
import json
import pandas as pd
import timm
import argparse
import gc
import torchvision
import yaml
import pickle
import wandb

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics import MAP
from waste_detector.training.dataset import (
    WasteImageDatasetNoMask,
    get_transforms
)
from waste_detector.training.config import Config, MODELS_FUNCTIONS
from waste_detector.training.utils import (
    fix_all_seeds,
    annotations_to_device,
    get_box_class_and_total_loss,
)
from torch.utils.data import DataLoader

def train_step(model, train_loader, config, scheduler, optimizer, n_batches):
    total_loss_accum, class_loss_accum, box_loss_accum = 0.0, 0.0, 0.0
        
    for batch_idx, (images, targets) in enumerate(train_loader, 1):
        # Predict
        images = images.to(config.DEVICE)        
        targets = annotations_to_device(targets, config.DEVICE)
        
        model.train()
        loss_dict = model(images, targets)
        #print(loss_dict.keys())
        #loss = loss_dict['loss']
        total_loss, class_loss, box_loss = get_box_class_and_total_loss(loss_dict)
    
        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_accum += total_loss.item()
        class_loss_accum += class_loss.item()
        box_loss_accum += box_loss.item()
        
        gc.collect()
        
    scheduler.step()

    return model, total_loss_accum, class_loss_accum, box_loss_accum

def val_step(model, val_loader, config, n_batches_val):
    # If the model is set up to eval, it does not return losses
    total_loss_accum, class_loss_accum, box_loss_accum = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader, 1):
            images = images.to(config.DEVICE)
            targets = annotations_to_device(targets, config.DEVICE)
            
            model.train()
            val_loss_dict = model(images, targets)
            total_loss, class_loss, box_loss = get_box_class_and_total_loss(val_loss_dict)

            total_loss_accum += total_loss.item()
            class_loss_accum += class_loss.item()
            box_loss_accum += box_loss.item()
            
            gc.collect()

    return model, total_loss_accum, class_loss_accum, box_loss_accum

def fit(model, train_loader, val_loader, config, filepath):
#     for param in model.parameters():
#         param.requires_grad = True

    model = model.to(config.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.LEARNING_RATE,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)
    
    n_batches, n_batches_val = len(train_loader), len(val_loader)

    model.train()

    best_loss = np.inf
    val_loss_accum, train_loss_accum = [], []

    with torch.cuda.device(config.DEVICE):
        for epoch in range(1, config.EPOCHS + 1):
            model, train_loss, train_class_loss, train_box_loss = train_step(model,
                                                                             train_loader,
                                                                             config,
                                                                             lr_scheduler,
                                                                             optimizer,
                                                                             n_batches)
            
            train_loss = train_loss / n_batches
            train_class_loss = train_class_loss / n_batches
            train_box_loss = train_box_loss / n_batches
            train_loss_accum.append(train_loss)
            
            gc.collect()

            model, val_loss, val_class_loss, val_box_loss = val_step(model,
                                                                       val_loader,
                                                                       config,
                                                                       n_batches_val)
            
            val_loss = val_loss / n_batches_val
            val_class_loss = val_class_loss / n_batches
            val_box_loss = val_box_loss / n_batches
            val_loss_accum.append(train_loss)

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            gc.collect()

            prefix = f"[Epoch {epoch:2d} / {config.EPOCHS:2d}]"
            print(prefix)
            print(f"{prefix} Train class loss: {train_class_loss:7.3f}. Val class loss: {val_class_loss:7.3f}")
            print(f"{prefix} Train box loss: {train_box_loss:7.3f}. Val box loss: {val_box_loss:7.3f}")
            print(f"{prefix} Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")

            if val_loss < best_loss:
                best_loss = val_loss
                print(f'{prefix} Save Val loss: {val_loss:7.3f}')
                torch.save(model.state_dict(), filepath)
                
            print(prefix)

    return model, train_loss_accum, val_loss_accum

def get_loaders(df_train, df_val, collate_fn, config=Config):
    ds_train = WasteImageDatasetNoMask(df_train, get_transforms(config, augment=True), config)
    dl_train = DataLoader(ds_train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn)

    ds_val = WasteImageDatasetNoMask(df_val, get_transforms(config, augment=False), config)
    dl_val = DataLoader(ds_val,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=collate_fn)

    return dl_train, dl_val

def train(parameters : Dict):
    fix_all_seeds(4444)
    
    with open(parameters['train_set'], 'rb') as file:
        train_df = pickle.load(file)
        
    with open(parameters['val_set'], 'rb') as file:
        val_df = pickle.load(file)

    functions = MODELS_FUNCTIONS[parameters['model']]
    train_loader, val_loader = get_loaders(train_df,
                                           val_df,
                                           functions['collate_fn'])
    print('Getting the model')
    model_function = functions['model_fn']
    params = functions['params']
    model = model_function(7, **params)

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
    model, train_loss, val_loss = fit(model,
                                      train_loader,
                                      val_loader,
                                      Config,
                                      parameters['checkpoint'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    train(params)