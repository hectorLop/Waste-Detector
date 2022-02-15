from io import BytesIO
from typing import Dict, Tuple, Union
from icevision import *
from icevision.models.checkpoint import model_from_checkpoint
from deployment.classifier import transform_image
from icevision.models import ross

import PIL
import torch
import numpy as np
import torchvision

MODEL_TYPE = ross.efficientdet

def predict_boxes(det_model : torch.nn.Module, image : Union[str, BytesIO],
            detection_threshold : float) -> Dict:
    """
    Make a prediction with the detection model.

    Args:
        det_model (torch.nn.Module): Detection model
        image (Union[str, BytesIO]): Image filepath if the image is one of
            the example images and BytesIO if the image is a custom image
            uploaded by the user.
        detection_threshold (float): Detection threshold

    Returns:
        Dict: Prediction dictionary.
    """        
    # Class map and transforms
    class_map = ClassMap(classes=['Waste'])
    transforms = tfms.A.Adapter([
                    *tfms.A.resize_and_pad(512),
                    tfms.A.Normalize()
                ])
    
    # Single prediction
    pred_dict  = MODEL_TYPE.end2end_detect(image,
                                           transforms, 
                                           det_model,
                                           class_map=class_map,
                                           detection_threshold=detection_threshold,
                                           return_as_pil_img=False,
                                           return_img=True,
                                           display_bbox=False,
                                           display_score=False,
                                           display_label=False)

    return pred_dict

def prepare_prediction(pred_dict : Dict,
                       nms_threshold : str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Get the predictions in a right format.

    Args:
        pred_dict (Dict): Prediction dictionary.
        nms_threshold (float): Threshold for the NMS postprocess.

    Returns:
        Tuple: Tuple containing the following:
            - (torch.Tensor): Bounding boxes
            - (np.ndarray): Image data
    """
    # Convert each box to a tensor and stack them into an unique tensor
    boxes = [box.to_tensor() for box in pred_dict['detection']['bboxes']]
    boxes = torch.stack(boxes)

    # Get the scores and labels as tensor
    scores = torch.as_tensor(pred_dict['detection']['scores'])
    labels = torch.as_tensor(pred_dict['detection']['label_ids'])

    image = np.array(pred_dict['img'])

    # Apply NMS to postprocess the bounding boxes
    fixed_boxes = torchvision.ops.batched_nms(boxes, scores,
                                              labels,nms_threshold)
    boxes = boxes[fixed_boxes, :]

    return boxes, image

def predict_class(classifier : torch.nn.Module, image : np.ndarray,
                  bboxes : torch.Tensor) -> np.ndarray:
    """
    Predict the class of each detected object.

    Args:
        classifier (torch.nn.Module): Classifier model.
        image (np.ndarray): Image data.
        bboxes (torch.Tensor): Bounding boxes.

    Returns:
        np.ndarray: Array containing the predicted class for each object.
    """
    preds = []

    for bbox in bboxes:
        img = image.copy()
        bbox = np.array(bbox).astype(int)

        # Get the bounding box content
        cropped_img = PIL.Image.fromarray(img).crop(bbox)
        cropped_img = np.array(cropped_img)

        # Apply transformations to the cropped image
        tran_image = transform_image(cropped_img, 224)
        # Channels first
        tran_image = tran_image.transpose(2, 0, 1)
        tran_image = torch.as_tensor(tran_image, dtype=torch.float).unsqueeze(0)

        # Make prediction
        y_preds = classifier(tran_image)
        preds.append(y_preds.softmax(1).detach().numpy())

    preds = np.concatenate(preds).argmax(1)

    return preds
