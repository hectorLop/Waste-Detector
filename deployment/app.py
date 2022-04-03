import sys
import icevision
import wandb
import torch
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from typing import Tuple

from utils import encode, decode
from classifier import CustomEfficientNet, CustomViT
from model import predict_boxes, prepare_prediction, predict_class
from icevision.models.checkpoint import model_from_checkpoint


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
    detector_ckpt = f'model_dir/efficientDet_icevision_v9.ckpt'
    print('Loading the detection model', flush=True)
    checkpoint_and_model = model_from_checkpoint(
                                detector_ckpt, 
                                model_name='ross.efficientdet',
                                backbone_name='d1',
                                img_size=512,
                                classes=['Waste'],
                                revise_keys=[(r'^model\.', '')],
                                map_location='cpu')

    det_model = checkpoint_and_model['model']
    det_model.eval() 
    print('Loading the classifier model', flush=True)
 
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
print('Loaded models')
logger.info('LOADED MODELS')

def handler(event, context):
    try:
        logger.info('PRUEBA')
        logger.info(event)
        body = json.loads(event['body'])
 
        image = decode(body['image'])
        #image = decode(body)
        logger.info('DECODED IMAGE')

        #detection_threshold = float(body['detection_threshold'])
        #nms_threshold = float(body['nms_threshold'])
        detection_threshold, nms_threshold = 0.5, 0.5

        print('Predicting bounding boxes')
        pred_dict = predict_boxes(detector, image, detection_threshold)
        logger.info('PREDICTED BBOXES')

        print('Fixing the preds')
        boxes, image = prepare_prediction(pred_dict, nms_threshold)
        logger.info('FIXING PREDS')

        print('Predicting classes')
        labels = predict_class(classifier, image, boxes)
        logger.info('PREDICTING CLASSES')

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
