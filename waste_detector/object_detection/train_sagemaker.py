import argparse
from typing import Callable, Dict, Tuple

import yaml

from wandb_mv.versioner import Versioner 

from icevision.data.dataset import Dataset
from icevision.metrics import COCOMetric, COCOMetricType
from icevision.models.ross.efficientdet.lightning import ModelAdapter
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from waste_detector.object_detection.config_sagemaker import Config
from waste_detector.object_detection.models import EfficientDetModel, MetricsCallback
from waste_detector.object_detection.utils import (
    fix_all_seeds,
    get_splits,
    get_transforms,
    get_metrics,
    get_best_metric
)
from waste_detector.model_registry.utils import get_latest_version


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
    config = Config(parameters)
    fix_all_seeds(config.seed)

    train_dl, valid_dl = get_data_loaders(
        config.annotations, config.img_dir, config.indices
    )

    extra_args = {
        'img_size': config.img_size
    }

    print("Getting the model")
    model = config.model_type.model(
        backbone=config.backbone(pretrained=True),
        num_classes=config.num_classes,
        **extra_args
    )

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    
    wandb_logger = WandbLogger(project="waste_detector",
                               entity="hlopez",
                               config={
                                   "learning_rate": config.learning_rate,
                                   "weight_decay": config.weight_decay,
                                   "momentum": config.momentum,
                                   "batch_size": config.batch_size,
                                   "total_epochs": config.epochs,
                                   "img_size": config.img_size,
                               })
    
    latest_version = get_latest_version('detector', wandb_logger.experiment)
    new_version = int(latest_version) + 1

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_path,
        filename=f'{config.checkpoint_name}_v{new_version}',
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
        old_epochs = config.epochs
        config.epochs = 5

        lightning_model = warm_up(lightning_model, train_dl, valid_dl,
                                  config, checkpoint_callback)

        config.epochs = old_epochs

    print("TRAINING")
    for param in lightning_model.model.parameters():
        param.requires_grad = True
        
        
    trainer = Trainer(max_epochs=config.epochs, gpus=1,
                      callbacks=[checkpoint_callback, metrics_callback], logger=wandb_logger)
    trainer.fit(lightning_model, train_dl, valid_dl)
    
    metrics = get_metrics(trainer, MetricsCallback)
    best_metric = get_best_metric(metrics)
    
    versioner = Versioner(wandb_logger.experiment)
    
    artifact = versioner.create_artifact(
                                checkpoint=f'{config.checkpoint_path}/{config.checkpoint_name}_v{new_version}.ckpt',
                                artifact_name='detector',
                                artifact_type='model',
                                description='Prueba Wandb-MV',
                                metadata={
                                    'val_metric': best_metric,
                                    'test_metric': 0.0,
                                    'model_type': str(config.model_type).split("'")[1],
                                    'backbone': config.backbone.model_name,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotations", type=str)
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--indices", type=str)
    parser.add_argument("--chekpoint_path", type=str)
    parser.add_argument("--checkpoint_name", type=str)
    parser.add_argument("--warm_up", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--model_type", type=Callable) # Use string and cast to callable?
    parser.add_argument("--backbone", type=Callable) # Use string and cast to callable?
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.000001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    train(params)
