from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from icevision.models.checkpoint import model_from_checkpoint
from classifier import CustomViT

def plot_img_no_mask(image : np.ndarray, boxes : torch.Tensor, labels):
    colors = {
        0: (255,255,0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0,128,0),
        4: (255,165,0),
        5: (230,230,250),
        6: (192,192,192)
    }

    texts = {
        0: 'plastic',
        1: 'dangerous',
        2: 'carton',
        3: 'glass',
        4: 'organic',
        5: 'rest',
        6: 'other'
    }

    # Show image
    boxes = boxes.cpu().detach().numpy().astype(np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, box in enumerate(boxes):
        color = colors[labels[i]]

        [x1, y1, x2, y2] = np.array(box).astype(int)
        # Si no se hace la copia da error en cv2.rectangle
        image = np.array(image).copy()

        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv2.rectangle(image, pt1, pt2, color, thickness=5)
        cv2.putText(image, texts[labels[i]], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, thickness=5, color=color)

    plt.axis('off')
    ax.imshow(image)

    fig.savefig("img.png", bbox_inches='tight')

def get_models(
    detection_ckpt : str,
    classifier_ckpt : str
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Get the detection and classifier models

    Args:
        detection_ckpt (str): Detection model checkpoint
        classifier_ckpt (str): Classifier model checkpoint

    Returns:
        tuple: Tuple containing:
            - (torch.nn.Module): Detection model
            - (torch.nn.Module): Classifier model
    """
    print('Loading the detection model')
    checkpoint_and_model = model_from_checkpoint(
                                detection_ckpt,
                                model_name='ross.efficientdet',
                                backbone_name='d0',
                                img_size=512,
                                classes=['Waste'],
                                revise_keys=[(r'^model\.', '')])

    det_model = checkpoint_and_model['model']
    det_model.eval()

    print('Loading the classifier model')
    classifier = CustomViT(target_size=7, pretrained=False)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu'))
    classifier.eval()

    return det_model, classifier