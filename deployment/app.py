import sys
import icevision
import wandb
import torch
import json
import logging
import PIL
import pickle
import datetime
import boto3

from typing import Tuple, Dict

from utils import encode, decode, get_data_drift 
from classifier import CustomEfficientNet, CustomViT
from model import predict_boxes, prepare_prediction, predict_class
from icevision.models.checkpoint import model_from_checkpoint
import icevision.models as models


def load_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
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
    # Detector checkpoints and model config
    detector_ckpt = f'model_dir/efficientDet_icevision_v9.ckpt'
    extra_args = {}
    model_type = models.ross.efficientdet
    backbone = model_type.backbones.d1(pretrained=False)
    extra_args['img_size'] = 512

    # Create the detector model
    det_model = model_type.model(backbone=backbone, num_classes=2,
                                pretrained_backbone=False, **extra_args)

    # Load the detector checkpoint
    ckpt = torch.load(detector_ckpt, map_location=torch.device('cpu'))
    det_model.load_state_dict(ckpt)
    det_model.eval()

    # Classifier checkpoint and model creation
    classifier_ckpt = 'model_dir/class_efficientB0_taco_7_class_v1.pth'
    classifier = CustomEfficientNet(target_size=7, pretrained=False)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu'))
    classifier.eval()

    return det_model, classifier

def format_response(body, status_code):
    return {
        'statusCode': str(status_code),
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            }
        }

detector, classifier = load_models()

with open('model_dir/training_data_dist.pkl', 'rb') as file:
    data_dist = pickle.load(file)

def handler(event, context):
    try:
        body = event

        # Decode the image and get the NMS and detection thresholds
        image = decode(body['image'])
        detection_threshold = float(body['detection_threshold'])
        nms_threshold = float(body['nms_threshold'])

        hue, sat, brightness = get_data_drift(image, data_dist)

        cloudwatch = boto3.client('cloudwatch')

        try:
            response = cloudwatch.put_metric_data(
                MetricData=[
                        {
                            'MetricName': 'Hue',
                            'Unit': 'None',
                            'Value': hue
                        },
                        {
                            'MetricName': 'Saturation',
                            'Unit': 'None',
                            'Value': sat
                        },
                        {
                            'MetricName': 'Brightness',
                            'Unit': 'None',
                            'Value': brightness
                        }

                    ],
                Namespace='Waste-Detector-Drift'
            )
        except Exception as e:
            print(e)

        # Predict the bounding boxed
        pred_dict = predict_boxes(detector, image, detection_threshold)
        # Postprocess the predicted boundinf boxes using NMS 
        boxes, image = prepare_prediction(pred_dict, nms_threshold)

        # Predict the classes for each detected object
        labels = predict_class(classifier, image, boxes)

        # Convert the image to PIL and encode it
        image = PIL.Image.fromarray(image)
        image = encode(image)

        payload = {
            'image': image,
            'boxes': boxes.tolist(),
            'labels': labels.tolist()
        }

        return format_response(payload, 200)
    except:
        return format_response({'msg': 'ERROR'}, 200)
