import timm
import torch
import torch.nn as nn

class CustomEfficientNet(nn.Module):
    """
    This class defines a custom EfficientNet network.

    Parameters
    ----------
    target_size : int
        Number of units for the output layer.
    pretrained : bool
        Determine if pretrained weights are used.

    Attributes
    ----------
    model : nn.Module
        EfficientNet model.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        target_size: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Modify the classifier layer
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, target_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        return x
