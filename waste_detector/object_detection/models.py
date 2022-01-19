# import sys
# sys.path.insert(0, '/home/icevision/icevision/')
import copy

from icevision.models.ross.efficientdet.lightning import ModelAdapter
from torch.optim import SGD
from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.inference import *
from waste_detector.object_detection.config import Config
from pytorch_lightning.callbacks import Callback

class EfficientDetModel(ModelAdapter):
    """
    Lighting wrapper of a EfficientDet model.
    """
    
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=Config.LEARNING_RATE,
                   momentum=Config.MOMENTUM)

    
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {
            'valid/loss': [],
            'COCOMetric': []
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics
        
        self.metrics['valid/loss'].append(metric['valid/loss'].item())
        self.metrics['COCOMetric'].append(metric['COCOMetric'].item())