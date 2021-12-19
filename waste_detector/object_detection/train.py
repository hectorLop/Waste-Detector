from typing import Dict
from icevision.models.ross.efficientdet import lightning
from pytorch_lightning import callbacks
#from icevision.data import data_splitter
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

from waste_detector.object_detection.config import Config
from waste_detector.training.utils import (
    fix_all_seeds,
    annotations_to_device,
    get_box_class_and_total_loss,
)
from waste_detector.object_detection.models import EfficientDetModel

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from icevision.metrics import COCOMetric, COCOMetricType
from icevision.parsers.coco_parser import COCOBBoxParser
from icevision.data.data_splitter import RandomSplitter
from icevision.data.dataset import Dataset
import icevision.tfms as tfms

def get_splits(annotations, img_dir, config=Config):
    parser = COCOBBoxParser(annotations_filepath=annotations,
                            img_dir=img_dir)
    splitter = RandomSplitter(probs=config.PROBS,
                              seed=config.SEED)

    train, test, val = parser.parse(data_splitter=splitter,
                                    autofix=True)

    return train, test, val

def get_transforms(config):
    image_size = config.IMG_SIZE

    train_tfms = tfms.A.Adapter([
        *tfms.A.aug_tfms(size=image_size, presize=config.PRESIZE),
        tfms.A.Normalize()
    ])

    valid_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(image_size),
        tfms.A.Normalize()
    ])

    test_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(image_size),
        tfms.A.Normalize()
    ])

    return train_tfms, valid_tfms, test_tfms

def warm_up(lighting_model, train_dl, valid_dl, config, checkpoint_callback):
    for param in lighting_model.model.parameters():
        param.requires_grad = False
        
    for param in lighting_model.model.model.class_net.parameters():
        param.requires_grad = True
        
    for param in lighting_model.model.model.box_net.parameters():
        param.requires_grad = True
        
    trainer = Trainer(max_epochs=config.EPOCHS,
                      gpus=1,
                      callbacks=[checkpoint_callback])
    trainer.fit(lighting_model, train_dl, valid_dl)
    
    return trainer.model

def train(parameters : Dict):
    fix_all_seeds(Config.SEED)

    train_records, test_records, val_records = get_splits(parameters['annotations'], parameters['img_dir'])

    train_tfms, valid_tfms, test_tfms = get_transforms(Config)

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(val_records, valid_tfms)
    test_ds = Dataset(test_records, test_tfms)

    # Data Loaders
    train_dl = Config.MODEL_TYPE.train_dl(train_ds,
                                          batch_size=Config.BATCH_SIZE,
                                          num_workers=4,
                                          shuffle=True)
    valid_dl = Config.MODEL_TYPE.valid_dl(valid_ds,
                                          batch_size=Config.BATCH_SIZE,
                                          num_workers=4,
                                          shuffle=False)
    test_dl = Config.MODEL_TYPE.valid_dl(test_ds,
                                         batch_size=Config.BATCH_SIZE,
                                         num_workers=4,
                                         shuffle=False)
    print('Getting the model')
    model = Config.MODEL_TYPE.model(backbone=Config.BACKBONE(pretrained=True),
                                    num_classes=Config.NUM_CLASSES, 
                                    **Config.EXTRA_ARGS)
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

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    checkpoint_callback = ModelCheckpoint(dirpath=parameters['checkpoint_path'],
                                          filename=parameters['checkpoint_name'],
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='valid/loss',
                                          mode='min')

    lightning_model = EfficientDetModel(model=model,
                                        metrics=metrics)

    if parameters['warm_up']:
        print('WARMING_UP')
        warm_up_cfg = Config()
        warm_up_cfg.EPOCHS = 5
        # lightning_model = EfficientDetModel(model=model,
        #                                     metrics=metrics)
        lightning_model = warm_up(lightning_model, train_dl, valid_dl,
                                  warm_up_cfg, checkpoint_callback)
    
    print('TRAINING')
    for param in lightning_model.model.parameters():
        param.requires_grad = True

    trainer = Trainer(max_epochs=Config.EPOCHS,
                      gpus=1,
                      callbacks=[checkpoint_callback])
    trainer.fit(lightning_model, train_dl, valid_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    train(params)