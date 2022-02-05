from icevision.models.ross.efficientdet.lightning import ModelAdapter
from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.inference import *
from pytorch_lightning.callbacks import Callback

class EfficientDetModel(ModelAdapter):
    """
    Lighting wrapper of a EfficientDet model.
    """

    def __init__(self, model: nn.Module, optimizer, learning_rate, metrics = None, ):
        super().__init__(model, metrics)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)
        # return SGD(self.parameters(), lr=Config.learning_rate,
        #            momentum=Config.momentum)

    
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