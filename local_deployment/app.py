import icevision
import torch
import json
import PIL
import pickle
import datetime

from typing import Tuple, Dict
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

from utils import encode, decode
from classifier import CustomEfficientNet, CustomViT
from model import predict_boxes, prepare_prediction, predict_class
from icevision.models.checkpoint import model_from_checkpoint
import icevision.models as models

from huggingface_hub import hf_hub_url, cached_download, hf_hub_download


CLASSIFIER_REPO = 'hlopez/ViT_waste_classifier'
CLASSIFIER_FILENAME = 'classifier.ckpt'
DETECTOR_REPO = 'hlopez/EfficientDet_waste_detector'
DETECTOR_FILENAME = 'detector.ckpt'

app = FastAPI(
    title="Waste-Detector",
    description="Detect waste in images",
    version="0.1",
)

def load_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Get the detection and classifier models

    Returns:
        tuple: Tuple containing:
            - (torch.nn.Module): Detection model
            - (torch.nn.Module): Classifier model
    """
    #global detector, classifier

    detector_ckpt = hf_hub_download(
        repo_id=DETECTOR_REPO,
        filename=DETECTOR_FILENAME)
    classifier_ckpt = hf_hub_download(
        repo_id=CLASSIFIER_REPO,
        filename=CLASSIFIER_FILENAME)

    checkpoint_and_model = model_from_checkpoint(
                                detector_ckpt,
                                model_name='ross.efficientdet',
                                backbone_name='d1',
                                img_size=512,
                                classes=['Waste'],
                                revise_keys=[(r'^model\.', '')],
                                map_location='cpu')

    detector = checkpoint_and_model['model']
    detector.eval()

    classifier = CustomViT(target_size=7, pretrained=False)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu'))
    classifier.eval()

    return detector, classifier

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

print('READY TO MAKE PREDICTIONS')

@app.post('/predict', tags=['Prediction'])
def predict(image = Body(...), detection_threshold = Body(...), nms_threshold = Body(...)) -> Dict:
    ##try:
    print('LLEGA REQUEST')
    print(detection_threshold)

    # Decode the image and get the NMS and detection thresholds
    image = decode(image)
    #detection_threshold = float(body['detection_threshold'])
    #nms_threshold = float(body['nms_threshold'])
    print('HOLA')
    print(type(detection_threshold))
    print(type(nms_threshold))

    # Predict the bounding boxed
    pred_dict = predict_boxes(detector, image, detection_threshold)
    print('BOXES')
    # Postprocess the predicted boundinf boxes using NMS 
    boxes, image = prepare_prediction(pred_dict, nms_threshold)
    print('PREDICTION PREPARED')

    # Predict the classes for each detected object
    labels = predict_class(classifier, image, boxes)
    print(f'LABELSSS: {labels}')

    # Convert the image to PIL and encode it
    #image = PIL.Image.fromarray(image)
    #image = encode(image)

    payload = {
        #'image': image,
        'boxes': boxes.tolist(),
        'labels': labels.tolist()
    }

    return JSONResponse(content=payload, media_type='application/json')
##except Exception as e:
    #print(e)
    #print('ERRORRRRRR')
    #return format_response({'msg': 'ERROR'}, 200)
