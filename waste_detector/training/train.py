import torch
import numpy as np
import json
import pandas as pd
import timm
import argparse
import gc
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics import MAP
from waste_detector.training.dataset import (
    WasteImageDatasetNoMask,
    get_transforms
)
from waste_detector.training.config import Config
from waste_detector.training.utils import fix_all_seeds
from waste_detector.training.data_split import split_data
from waste_detector.training.models import get_custom_faster_rcnn, create_efficientdet_model
from waste_detector.categories_format.format import process_categories
from torch.utils.data import DataLoader

def train_step(model, train_loader, config, scheduler, optimizer, n_batches):
    loss_accum = 0.0
    
    preds, ground_truths = [], []
    
    for batch_idx, (images, targets) in enumerate(train_loader, 1):
        # Predict
        images = list(image.to(config.DEVICE).float() for image in images)
        ground_truths += targets
        targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
        
        model.train()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
    
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()
        model.eval()
        temp_preds = model(images)
        temp_preds = [{k: v.detach().cpu() for k, v in t.items()} for t in temp_preds]
        preds += temp_preds
        
        gc.collect()

    map_metric = MAP()
    map_metric.update(preds, ground_truths)
    result = map_metric.compute()
    map_value = result['map'].item()
        
    scheduler.step()

    return model, loss_accum, map_value

def val_step(model, val_loader, config, n_batches_val):
    # If the model is set up to eval, it does not return losses

    val_loss_accum = 0
    preds, ground_truths = [], []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader, 1):
            images = list(image.to(config.DEVICE).float() for image in images)
            ground_truths += targets
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
            
            model.train()
            val_loss_dict = model(images, targets)
            val_batch_loss = sum(loss for loss in val_loss_dict.values())

            val_loss_accum += val_batch_loss.item()
            
            gc.collect()
            
            model.eval()
            temp_preds = model(images)
            temp_preds = [{k: v.detach().cpu() for k, v in t.items()} for t in temp_preds]
            preds += temp_preds
    
    map_metric = MAP()
    map_metric.update(preds, ground_truths)
    result = map_metric.compute()
    map_value = result['map'].item()
    # Validation losses
    val_loss = val_loss_accum / n_batches_val

    return model, val_loss_accum, map_value

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
            model, loss, train_map = train_step(model,
                                                train_loader,
                                                config,
                                                lr_scheduler,
                                                optimizer,
                                                n_batches)
            
            train_loss = loss / n_batches
            train_loss_accum.append(train_loss)
            
            gc.collect()

            model, loss, val_map = val_step(model,
                                            val_loader,
                                            config,
                                            n_batches_val)
            
            val_loss = loss / n_batches_val
            val_loss_accum.append(train_loss)
            
            gc.collect()

            prefix = f"[Epoch {epoch:2d} / {config.EPOCHS:2d}]"
            print(prefix)
            print(f"{prefix} Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f}")
            print(f"{prefix} Train mAP: {train_map:7.3f}. Val mAP: {val_map:7.3f}")

            if val_loss < best_loss:
                best_loss = val_loss
                print(f'{prefix} Save Val loss: {val_loss:7.3f}')
                torch.save(model.state_dict(), filepath)
                
            print(prefix)

    return model, train_loss_accum, val_loss_accum

def get_loaders(df_train, df_val, config=Config):
    ds_train = WasteImageDatasetNoMask(df_train, get_transforms(augment=True), config)
    dl_train = DataLoader(ds_train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=lambda x: tuple(zip(*x)))

    ds_val = WasteImageDatasetNoMask(df_val, get_transforms(augment=False), config)
    dl_val = DataLoader(ds_val,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=lambda x: tuple(zip(*x)))

    return dl_train, dl_val

def aggregate_datasets(annotations_df, images_df):
    data = []

    for id in annotations_df['image_id']:
        image_data = images_df[images_df['id'] == id]
        
        file_name = image_data['file_name'].values[0]
        width= image_data['width'].values[0]
        height = image_data['height'].values[0]

        data.append((file_name, width, height))

    df = pd.DataFrame(data, columns=['filename', 'width', 'height'])
    annotations_df = pd.concat([annotations_df, df], axis=1)

    return annotations_df

def get_efficientnet_model(num_classes):
    efficientnet = timm.create_model('efficientnet_b0', pretrained=True)

    backbone = torch.nn.Sequential(
        efficientnet.conv_stem,
        efficientnet.bn1,
        efficientnet.act1,
        efficientnet.blocks,
        efficientnet.conv_head,
        efficientnet.bn2,
        efficientnet.act2
    )

    out_channels = 1280
    model = get_custom_faster_rcnn(backbone, out_channels, num_classes+1, Config)

    return model

def get_faster_rcnn(num_classes, config=Config):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    return model

def get_efficientnetv2_model(num_classes):
    efficientnet = timm.create_model('efficientnetv2_rw_s', pretrained=True)

    backbone = torch.nn.Sequential(
        efficientnet.conv_stem,
        efficientnet.bn1,
        efficientnet.act1,
        efficientnet.blocks,
        efficientnet.conv_head,
        efficientnet.bn2,
        efficientnet.act2
    )

    out_channels = 1792
    model = get_custom_faster_rcnn(backbone, out_channels, num_classes+1, Config)

    return model

def train(annotations_path):
    fix_all_seeds(4444)

    with open(annotations_path, 'r') as file:
        annotations = json.load(file)

    categories_df = pd.DataFrame(annotations['categories'])
    images_df = pd.DataFrame(annotations['images'])
    annotations_df = pd.DataFrame(annotations['annotations'])

    print('Aggregating the datasets')
    annotations_df = aggregate_datasets(annotations_df, images_df)
    print('Processing the new categories')
    annotations_df, categories_df = process_categories(categories_df,
                                                       annotations_df)

    print('Preparing the data')
    train_df, val_df, test_df = split_data(annotations_df)
    train_loader, val_loader = get_loaders(train_df, val_df)
    print('Getting the model')
    #model = get_efficientnet_model(7)
    #model = get_faster_rcnn(7)
    model = create_efficientdet_model(7, 512, 'efficientdet_d0')
    print('TRAINING')
    model, train_loss, val_loss = fit(model, train_loader, val_loader, Config, 'efficientdet.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help='annotations JSON')
    args = parser.parse_args()

    train(args.file)