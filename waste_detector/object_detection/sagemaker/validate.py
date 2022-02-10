import argparse
from typing import Dict, Tuple

import torch
import icevision
import wandb
import yaml
import glob

from wandb_mv.versioner import Versioner 

from icevision.data.dataset import Dataset
from icevision.metrics import COCOMetric, COCOMetricType
from icevision.models.ross.efficientdet.lightning import ModelAdapter
from icevision.models.checkpoint import model_from_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models import EfficientDetModel, MetricsCallback
from utils import (
    fix_all_seeds,
    get_test_split,
    get_transforms,
    get_metrics,
    get_best_metric
)

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
    test_dl = model_type.valid_dl(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False
    )

    return test_dl


def validate(config : Dict) -> None:
    """
    Trains a waste detector model.

    Args:
        parameters (Dict): Dictionary containing training parameters.
    """
    fix_all_seeds(int(config['seed']))

    model_type = get_object_from_str(config['model_type'])
    backbone = get_object_from_str(config['backbone'])

    test_dl = get_data_loaders(model_type, config)

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

    run = wandb.init(project="waste_detector", entity="hlopez",)

    best_model_art = run.use_artifact('detector:best_model')
    model_path = best_model_art.download('models/')
    model_ckpt = glob.glob(f'{model_path}*')[0]

    checkpoint_and_model = model_from_checkpoint(
                                model_ckpt,
                                model_name='ross.efficientdet',
                                backbone_name='d1',
                                img_size=512,
                                classes=['Waste'],
                                revise_keys=[(r'^model\.', '')],
                                map_location='cpu')

    model = checkpoint_and_model['model']
    model = model.to('cpu')
    model.eval()

    metrics_callback = MetricsCallback()
    lightning_model = EfficientDetModel(model=model, metrics=metrics)
        
    trainer = Trainer(max_epochs=int(config['epochs']), gpus=1,
                      callbacks=[metrics_callback])

    trainer.validate(lightning_model, test_dl)
    
    metrics = get_metrics(trainer, MetricsCallback)
    test_metric = get_best_metric(metrics)

    best_model_art.metadata['test_metric'] = test_metric

    versioner = Versioner(run)
    versioner.promote_model(new_model=best_model_art,
                            artifact_name='detector',
                            artifact_type='model',
                            comparision_metric='test_metric',
                            promotion_alias='production',
                            comparision_type='smaller',
                            already_deployed=True
                           )

if __name__ == "__main__":
    with open('/opt/ml/input/config/hyperparameters.json', 'r') as file:
        params = json.load(file)

    validate(params)
