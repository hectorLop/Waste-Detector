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
    get_test_split,
    get_transforms,
    get_metrics,
    get_best_metric
)
from waste_detector.model_registry.utils import promote_to_production


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
    test_records = get_test_split(annotations, img_dir, indices)
    # Training, validation and test transforms
    _, _, test_tfms = get_transforms(config)

    # Datasets
    test_ds = Dataset(test_records, test_tfms)

    # Data Loaders
    test_dl = config.MODEL_TYPE.valid_dl(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False
    )

    return test_dl


def validate(parameters: Dict) -> None:
    """
    Trains a waste detector model.

    Args:
        parameters (Dict): Dictionary containing training parameters.
    """
    fix_all_seeds(Config.SEED)

    test_dl = get_data_loaders(
        parameters["annotations"], parameters["img_dir"], parameters['indices']
    )

    print("Getting the model")
    model = Config.MODEL_TYPE.model(
        backbone=Config.BACKBONE(pretrained=False),
        num_classes=Config.NUM_CLASSES,
        **Config.EXTRA_ARGS
    )

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    run = wandb.init(project="waste_detector", entity="hlopez",)

    best_model_art = run.experiment.use_artifact('detector:best_model')
    model_path = best_model_art.download('models/')
    model.load_state_dict(torch.load(model_path))
    
    
    metrics_callback = MetricsCallback()
    lightning_model = EfficientDetModel(model=model, metrics=metrics)
        
    trainer = Trainer(max_epochs=Config.EPOCHS, gpus=1,
                      callbacks=[metrics_callback])

    trainer.validate(lightning_model, test_dl)
    
    metrics = get_metrics(trainer, MetricsCallback)
    test_metric = get_best_metric(metrics)

    best_model_art.metadata['test_metric'] = test_metric
    
    promote_to_production(best_model_art, 'detector', run)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Config YAML file")
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    validate(params)