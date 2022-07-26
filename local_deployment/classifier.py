import timm
import torch.nn as nn
import albumentations as A
import torch
import cv2

class CustomNormalization(A.ImageOnlyTransform):
    def _norm(self, img):
        return img / 255.

    def apply(self, img, **params):
        return self._norm(img)

def transform_image(image, size):
    transforms = [
        A.Resize(size, size,
                interpolation=cv2.INTER_NEAREST),
        CustomNormalization(p=1),
    ]

    augs = A.Compose(transforms)
    transformed = augs(image=image)

    return transformed['image']

class CustomViT(nn.Module):
    """
    This class defines a custom ViT network.

    Parameters
    ----------
    target_size : int
        Number of units for the output layer.
    pretrained : bool
        Determine if pretrained weights are used.

    Attributes
    ----------
    model : nn.Module
        CustomViT model.
    """
    def __init__(self, model_name : str = 'vit_base_patch16_224',
                 target_size : int = 7, pretrained : bool = True):
        super().__init__()
        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       num_classes=target_size)

        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, target_size)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        return x
