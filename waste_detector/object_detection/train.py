import argparse
from typing import Dict, Tuple

import torch
import icevision
import wandb
import yaml
#import sys
#sys.path.insert(0, '../../../icevision/icevision/')

from icevision.data.dataset import Dataset
from icevision.metrics import COCOMetric, COCOMetricType
from icevision.models.ross.efficientdet.lightning import ModelAdapter
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from waste_detector.object_detection.config import Config
from waste_detector.object_detection.models import EfficientDetModel, MetricsCallback
from waste_detector.object_detection.utils import (
    fix_all_seeds,
    get_splits,
    get_transforms,
    get_metrics,
    get_best_metric
)
from waste_detector.model_registry.utils import publish_model, promote_to_best_model, get_latest_version


def warm_up(
    lighting_model: ModelAdapter,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    config: Config,
    checkpoint_callback: ModelCheckpoint,
) -> ModelAdapter:
    """
    Fine tune the classes and boxes heads.

    Args:
        lighting_model (ModelAdapter): PytorchLighting model.
        train_dl (DataLoader): Training dataloader.
        valid_dl (DataLoader): Validation dataloader.
        config (Config): Configuration object.
        checkpoint_callback (ModelCheckpoint): Callback to save checkpoints.

    Returns:
        (ModelAdapter): Fine tuned model.
    """
    for param in lighting_model.model.parameters():
        param.requires_grad = False

    for param in lighting_model.model.model.class_net.parameters():
        param.requires_grad = True

    for param in lighting_model.model.model.box_net.parameters():
        param.requires_grad = True

    trainer = Trainer(max_epochs=config.EPOCHS, gpus=1,
                      callbacks=[checkpoint_callback])
    trainer.fit(lighting_model, train_dl, valid_dl)

    return trainer.model


def get_data_loaders(
    annotations: str, img_dir: str,  indices : Dict, config: Config = Config
) -> Tuple[DataLoader]:
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
    train_records, val_records = get_splits(annotations, img_dir, indices)
    # Training, validation and test transforms
    train_tfms, valid_tfms, _ = get_transforms(config)

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(val_records, valid_tfms)
    #test_ds = Dataset(test_records, test_tfms)

    # Data Loaders
    train_dl = config.MODEL_TYPE.train_dl(
        train_ds, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=True
    )
    valid_dl = config.MODEL_TYPE.valid_dl(
        valid_ds, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False
    )
#     test_dl = config.MODEL_TYPE.valid_dl(
#         test_ds, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False
#     )

    return train_dl, valid_dl


def train(parameters: Dict) -> None:
    """
    Trains a waste detector model.

    Args:
        parameters (Dict): Dictionary containing training parameters.
    """
    fix_all_seeds(Config.SEED)

    train_dl, valid_dl = get_data_loaders(
        parameters["annotations"], parameters["img_dir"], parameters['indices']
    )

    print("Getting the model")
    model = Config.MODEL_TYPE.model(
        backbone=Config.BACKBONE(pretrained=True),
        num_classes=Config.NUM_CLASSES,
        **Config.EXTRA_ARGS
    )

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    
    wandb_logger = WandbLogger(project="waste_detector",
                               entity="hlopez",
                               config={
                                   "learning_rate": Config.LEARNING_RATE,
                                   "weight_decay": Config.WEIGHT_DECAY,
                                   "momentum": Config.MOMENTUM,
                                   "batch_size": Config.BATCH_SIZE,
                                   "total_epochs": Config.EPOCHS,
                                   "img_size": Config.IMG_SIZE,
                               })
    
    latest_version = get_latest_version('detector', wandb_logger.experiment)
    new_version = int(latest_version) + 1
    print(f'{parameters["checkpoint_name"]}_v{new_version}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=parameters["checkpoint_path"],
        filename=f'{parameters["checkpoint_name"]}_v{new_version}',
        save_top_k=1,
        verbose=True,
        monitor="valid/loss",
        mode="min",
    )
    
    metrics_callback = MetricsCallback()
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    lightning_model = EfficientDetModel(model=model, metrics=metrics)
    
    if parameters["warm_up"]:
        print("WARMING_UP")
        warm_up_cfg = Config()
        warm_up_cfg.EPOCHS = 5

        lightning_model = warm_up(lightning_model, train_dl, valid_dl,
                                  warm_up_cfg, checkpoint_callback)

    print("TRAINING")
    for param in lightning_model.model.parameters():
        param.requires_grad = True
        
        
    trainer = Trainer(max_epochs=Config.EPOCHS, gpus=1,
                      callbacks=[checkpoint_callback, metrics_callback], logger=wandb_logger)
    trainer.fit(lightning_model, train_dl, valid_dl)
    
    metrics = get_metrics(trainer, MetricsCallback)
    best_metric = get_best_metric(metrics)
    
    artifact = publish_model(checkpoint=f'{parameters["checkpoint_path"]}/{parameters["checkpoint_name"]}_v{new_version}.ckpt',
                              metric=best_metric,
                              model_type=str(Config.MODEL_TYPE).split("'")[1],
                              backbone=Config.BACKBONE.model_name,
                              extra_args=Config.EXTRA_ARGS,
                              name='detector',
                              run=wandb_logger.experiment)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Config YAML file")
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    train(params)
