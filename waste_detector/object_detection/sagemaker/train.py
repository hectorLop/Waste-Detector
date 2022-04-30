from typing import Dict, Tuple
import json
import glob
import sys
from wandb_mv.versioner import Versioner 

from icevision.data.dataset import Dataset
from icevision.metrics import COCOMetric, COCOMetricType
from icevision.models.ross.efficientdet.lightning import ModelAdapter
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch

from models import EfficientDetModel, MetricsCallback
from utils import (
    fix_all_seeds,
    get_splits,
    get_transforms,
    get_metrics,
    get_best_metric,
    get_object_from_str
)

def get_data_loaders(model_type, config) -> Tuple[DataLoader]:
    """
    Get the dataloaders for each set.

    Args:
        annotations (str): Annotations filepath.
        img_dir (str): Images filepath.
        config (Config): Config object.

    Returns:
        Tuple[DataLoader]: Tuple containing:
            - (DataLoader): Training dataloader
            - (DataLoader): Validation dataloader
            - (DataLoader): Test dataloader
    """
    # Training, test and validation records
    train_records, val_records = get_splits()
    # Training, validation and test transforms
    train_tfms, valid_tfms, _ = get_transforms(config)

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(val_records, valid_tfms)

    # Data Loaders
    train_dl = model_type.train_dl(
        train_ds, batch_size=int(config['batch_size']), num_workers=4, shuffle=True
    )
    valid_dl = model_type.valid_dl(
        valid_ds, batch_size=int(config['batch_size']), num_workers=4, shuffle=False
    )

    return train_dl, valid_dl

def train(config: Dict) -> None:
    """
    Trains a waste detector model.

    Args:
        parameters (Dict): Dictionary containing training parameters.
    """
    fix_all_seeds(int(config['seed']))
    
    model_type = get_object_from_str(config['model_type'])
    backbone = get_object_from_str(config['backbone'])

    train_dl, valid_dl = get_data_loaders(model_type, config)

    extra_args = {
        'img_size': int(config['img_size'])
    }
    
    print("Getting the model")
    model = model_type.model(
        backbone=backbone(pretrained=True),
        num_classes=int(config['num_classes']),
        **extra_args
    )

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    
    wandb_logger = WandbLogger(project="waste_detector",
                               entity="hlopez",
                               config={
                                   "learning_rate": float(config['learning_rate']),
                                   "weight_decay": float(config['weight_decay']),
                                   "momentum": float(config['momentum']),
                                   "batch_size": int(config['batch_size']),
                                   "total_epochs": int(config['epochs']),
                                   "img_size": int(config['img_size']),
                               })

    versioner = Versioner(wandb_logger.experiment)
    latest_version = versioner.get_latest_version('detector')
    new_version = int(latest_version) + 1
    print(f'NEW FILENAME: detector_{new_version}')

    checkpoint_callback = ModelCheckpoint(
        dirpath='/opt/ml/model/',
        filename=f'detector_{new_version}',
        save_top_k=1,
        verbose=True,
        monitor="valid/loss",
        mode="min",
    )
    
    metrics_callback = MetricsCallback()

    pytorch_lightning_model = get_object_from_str(config['pytorch_lightning_model'])
    lightning_model = pytorch_lightning_model(model=model, optimizer=torch.optim.SGD,
                                              learning_rate=float(config['learning_rate']),
                                              metrics=metrics)
    
    print("TRAINING")
    for param in lightning_model.model.parameters():
        param.requires_grad = True
        
    trainer = Trainer(max_epochs=int(config['epochs']), gpus=1,
                      callbacks=[checkpoint_callback, metrics_callback], logger=wandb_logger)
    trainer.fit(lightning_model, train_dl, valid_dl)
    
    metrics = get_metrics(trainer, MetricsCallback)
    best_metric = get_best_metric(metrics)
    
    artifact = versioner.create_artifact(
                                checkpoint=f'/opt/ml/model/detector_{new_version}.ckpt',
                                artifact_name='detector',
                                artifact_type='model',
                                description='Prueba Wandb-MV',
                                metadata={
                                    'val_metric': best_metric,
                                    'test_metric': 0.0,
                                    'model_type': config['model_type'],
                                    'backbone': config['backbone'],
                                    'lightning_model': config['pytorch_lightning_model'],
                                    'extra_args': extra_args,
                                }
                )
    
    versioner.promote_model(new_model=artifact,
                            artifact_name='detector',
                            artifact_type='model',
                            comparision_metric='val_metric',
                            promotion_alias='best_model',
                            comparision_type='smaller'
                           )

    print('Training complete')

if __name__ == "__main__":
    with open('/opt/ml/input/config/hyperparameters.json', 'r') as json_file:
        hyperparameters = json.load(json_file)

    train(hyperparameters)
    sys.exit(0)
