import sys
import icevision
import wandb
import torch

from typing import Tuple
from utils import encode, decode

from utils import encode, decode
from classifier import CustomEfficientNet, CustomViT
from model import predict_boxes, prepare_prediction, predict_class
from icevision.models.checkpoint import model_from_checkpoint


def get_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
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
    detector_run = wandb.init(project="waste_detector", entity="hlopez",)

    best_model_art = detector_run.use_artifact('detector:production')
    model_path = best_model_art.download('.')
    detector_ckpt = f'efficientDet_icevision_v9.ckpt'
    print('Loading the detection model')
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
    
    print('Loading the classifier model')
    wandb.finish()
    classifier_run = wandb.init(project="waste_classifier", entity="hlopez",)

    best_model_art = classifier_run.use_artifact('classifier:production')
    model_path = best_model_art.download('.')
    classifier_ckpt = 'class_efficientB0_taco_7_class_v1.pth' 

    classifier = CustomEfficientNet(target_size=7, pretrained=False)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu'))
    classifier.eval()

    wandb.finish()

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

def handler(event, context):
    try:
        body = json.loads(event['body'])
        image = decode(body['image'])

        detection_threshold = float(body['detection_threshold'])
        nms_threshold = float(body['nms_threshold'])

        print('Predicting bounding boxes')
        pred_dict = predict_boxes(detector, image, detection_threshold)

        print('Fixing the preds')
        boxes, image = prepare_prediction(pred_dict, nms_threshold)

        print('Predicting classes')
        labels = predict_class(classifier, image, boxes)

        image = PIL.Image.fromarray(image)
        image = encode(image)

        payload = {
            'image': image,
            'boxes': boxes.tolist(),
            'labels': labels.tolist()
        }

        return format_response(payload, 200)
    except:
        body = {}

        return format_response(body, 200)

#def handler(event, context):
#    return f'Hello from AWS Lambda using Python {sys.version} and Icevision {icevision.__version__}'
