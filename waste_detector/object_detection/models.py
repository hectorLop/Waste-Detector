from icevision.models.ross.efficientdet.lightning import ModelAdapter
from torch.optim import SGD

from waste_detector.object_detection.config import Config


class EfficientDetModel(ModelAdapter):
    """
    Lighting wrapper of a EfficientDet model.
    """

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=Config.LEARNING_RATE,
                   momentum=Config.MOMENTUM)
