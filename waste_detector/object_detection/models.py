from icevision.models.ross.efficientdet.lightning import ModelAdapter
from waste_detector.object_detection.config import Config
from torch.optim import SGD

class EfficientDetModel(ModelAdapter):
    """
    Lighting wrapper of a EfficientDet model.
    """
    def configure_optimizers(self):
        return SGD(self.parameters(),
                   lr=Config.LEARNING_RATE,
                   momentum=Config.MOMENTUM)