import torch
from waste_detector.object_detection.train import get_data_loaders
from waste_detector.object_detection.config import Config
from waste_detector.object_detection.models import EfficientDetModel
from tests import ANNOTATIONS, IMG_DIR, INDICES

from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from icevision.metrics.coco_metric import COCOMetric, COCOMetricType

def test_get_data_loaders():
    """
    Check the get_data_loaders() function
    """
    train_dl, valid_dl = get_data_loaders(annotations=ANNOTATIONS,
                                          img_dir=IMG_DIR,
                                          indices=INDICES,
                                          config=Config)

    assert (train_dl and valid_dl)
    assert (isinstance(train_dl, DataLoader) and 
            isinstance(valid_dl, DataLoader))

def test_model_creation():
    """
    Test the model creation
    """
    model = Config.model_type.model(
        backbone=Config.backbone(pretrained=False),
        num_classes=Config.num_classes,
        **Config.extra_args
    )

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    lightning_model = EfficientDetModel(model=model, metrics=metrics)

    assert lightning_model

def test_train_epoch():
    """
    Test the model is capable of training
    """
    train_dl, valid_dl = get_data_loaders(annotations=ANNOTATIONS,
                                          img_dir=IMG_DIR,
                                          indices=INDICES,
                                          config=Config)

    model = Config.model_type.model(
        backbone=Config.backbone(pretrained=False),
        num_classes=Config.num_classes,
        **Config.extra_args
    )
    lightning_model = EfficientDetModel(model=model)

    # Clone old parameters
    old_parameters = [param.clone() for param in model.parameters()]

    # Train 1 epoch
    trainer = Trainer(max_epochs=1, gpus=0)
    trainer.fit(lightning_model, train_dl, valid_dl)

    # Check that the model paramters changed
    new_parameters = [param for param in model.parameters()]
    equality = [torch.equal(new_parameters[i], old_parameters[i]) for i in range(len(new_parameters))]

    assert False in equality