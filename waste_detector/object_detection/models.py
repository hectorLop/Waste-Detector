import icevision.models as models

from waste_detector.object_detection.config import Config
from torch.optim import SGD

class EfficientDetModel(models.ross.efficientdet.lightning.ModelAdapter):
    def configure_optimizers(self):
        return SGD(self.parameters(),
                   lr=Config.LEARNING_RATE,
                   momentum=Config.MOMENTUM)