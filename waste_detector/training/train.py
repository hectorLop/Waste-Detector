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

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics import MAP
from waste_detector.training.dataset import (
    WasteImageDatasetNoMask,
    get_transforms
)
from waste_detector.training.config import Config, MODELS_FUNCTIONS
from waste_detector.training.utils import fix_all_seeds, annotations_to_device
#from waste_detector.training.models import get_custom_faster_rcnn, create_efficientdet_model
from torch.utils.data import DataLoader

def train_step(model, train_loader, config, scheduler, optimizer, n_batches):
    loss_accum = 0.0
    
    preds, ground_truths = [], []
    
    for batch_idx, (images, targets) in enumerate(train_loader, 1):
        # Predict
        images = images.to(config.DEVICE)
        ground_truths += targets
        
        targets = annotations_to_device(targets, config.DEVICE)
        #targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
        
        model.train()
        loss_dict = model(images, targets)
        loss = loss_dict['loss']
        #loss = sum(loss for loss in loss_dict.values())
    
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()
        #model.eval()
        #temp_preds = model(images)
        #temp_preds = [{k: v.detach().cpu() for k, v in t.items()} for t in temp_preds]
        #preds += temp_preds
        
        gc.collect()

    #map_metric = MAP()
    #map_metric.update(preds, ground_truths)
    #result = map_metric.compute()
    #map_value = result['map'].item()
        
    scheduler.step()

    return model, loss_accum

def val_step(model, val_loader, config, n_batches_val):
    # If the model is set up to eval, it does not return losses

    val_loss_accum = 0
    preds, ground_truths = [], []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader, 1):
            #images = list(image.to(config.DEVICE).float() for image in images)
            images = images.to(config.DEVICE)
            ground_truths += targets
            #targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
            targets = annotations_to_device(targets, config.DEVICE)
            
            model.train()
            val_loss_dict = model(images, targets)
            val_batch_loss = val_loss_dict['loss']
            #val_batch_loss = sum(loss for loss in val_loss_dict.values())

            val_loss_accum += val_batch_loss.item()
            
            gc.collect()
            
            #model.eval()
            #temp_preds = model(images)
            #temp_preds = [{k: v.detach().cpu() for k, v in t.items()} for t in temp_preds]
            #preds += temp_preds
    
    #map_metric = MAP()
    #map_metric.update(preds, ground_truths)
    #result = map_metric.compute()
    #map_value = result['map'].item()

    return model, val_loss_accum

def fit(model, train_loader, val_loader, config, filepath):
#     for param in model.parameters():
#         param.requires_grad = True

    model = model.to(config.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.LEARNING_RATE,
                                momentum=0.9,
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
            model, loss = train_step(model,
                                     train_loader,
                                     config,
                                     lr_scheduler,
                                     optimizer,
                                     n_batches)
            
            train_loss = loss / n_batches
            train_loss_accum.append(train_loss)
            
            gc.collect()

            model, loss = val_step(model,
                                   val_loader,
                                   config,
                                   n_batches_val)
            
            val_loss = loss / n_batches_val
            val_loss_accum.append(train_loss)
            
            gc.collect()

            prefix = f"[Epoch {epoch:2d} / {config.EPOCHS:2d}]"
            print(prefix)
            print(f"{prefix} Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
            #print(f"{prefix} Train mAP: {train_map:7.3f}. Val mAP: {val_map:7.3f}")

            if val_loss < best_loss:
                best_loss = val_loss
                print(f'{prefix} Save Val loss: {val_loss:7.3f}')
                torch.save(model.state_dict(), filepath)
                
            print(prefix)

    return model, train_loss_accum, val_loss_accum

def get_loaders(df_train, df_val, collate_fn, config=Config):
    ds_train = WasteImageDatasetNoMask(df_train, get_transforms(augment=True), config)
    dl_train = DataLoader(ds_train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn)

    ds_val = WasteImageDatasetNoMask(df_val, get_transforms(augment=False), config)
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
        
    #train_df = pd.read_csv(parameters['train_set'])
    #val_df = pd.read_csv(parameters['val_set'])

    functions = MODELS_FUNCTIONS[parameters['model']]
    train_loader, val_loader = get_loaders(train_df,
                                           val_df,
                                           functions['collate_fn'])
    print('Getting the model')
    model_function = functions['model_fn']
    params = functions['params']
    model = model_function(7, **params)
    #model = get_efficientnet_model(7)
    #model = get_faster_rcnn(7)
    #model = create_efficientdet_model(7, 512, 'efficientdet_d0')
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