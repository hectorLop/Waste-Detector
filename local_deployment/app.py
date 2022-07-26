import torch
import PIL

from typing import Tuple, Dict
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, Response
from http import HTTPStatus

from utils import encode, decode
from classifier import CustomViT
from model import predict_boxes, prepare_prediction, predict_class
from icevision.models.checkpoint import model_from_checkpoint

from huggingface_hub import hf_hub_download


CLASSIFIER_REPO = 'hlopez/ViT_waste_classifier'
CLASSIFIER_FILENAME = 'classifier.ckpt'
DETECTOR_REPO = 'hlopez/EfficientDet_waste_detector'
DETECTOR_FILENAME = 'detector.ckpt'

app = FastAPI(
    title="Waste-Detector",
    description="Detect waste in images",
    version="0.1",
)

@app.on_event('startup')
def load_models() -> None:
    """Load the detection and classifier models."""
    global detector, classifier

    # Download both checkpoints from Huggingface Hub
    detector_ckpt = hf_hub_download(
        repo_id=DETECTOR_REPO,
        filename=DETECTOR_FILENAME)
    classifier_ckpt = hf_hub_download(
        repo_id=CLASSIFIER_REPO,
        filename=CLASSIFIER_FILENAME)

    # Load the detector
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

    # Load the classifier
    classifier = CustomViT(pretrained=False)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu'))
    classifier.eval()

@app.post('/predict', tags=['Prediction'])
def predict(
    image = Body(...),
    detection_threshold = Body(...),
    nms_threshold = Body(...)
) -> Response:
    """
    Inference endpoint.

    Args:
        image (Body): Image to use for inference.
        detection_threshold (Body): Threshold to filter bounding boxes.
        nms_threshold (Body): Threshold for the NMS postprocessing.

    Returns:
        Response: JSON data containing the predicted bounding boxes and
            labels.
    """
    try:
        # Decode the image and get the NMS and detection thresholds
        image = decode(image)

        # Predict the bounding boxed
        pred_dict = predict_boxes(detector, image, detection_threshold)
        # Postprocess the predicted boundinf boxes using NMS 
        boxes, image = prepare_prediction(pred_dict, nms_threshold)

        # Predict the classes for each detected object
        labels = predict_class(classifier, image, boxes)

        payload = {
            'boxes': boxes.tolist(),
            'labels': labels.tolist()
        }

        return JSONResponse(content=payload, media_type='application/json', status_code=200)
    except:
        raise HTTPException(status_code=500,
                        detail=f'An error occurred in the inference process')

@app.get('/')
def index() -> Dict:
    """Health check"""
    if classifier and detector:
        response = {
            'message': 'Application ready for inference',
            'status-code': HTTPStatus.OK,
            'data': {},
        }
    else:
        message = 'The application is not ready for inference at the moment, wait please.'
        response = {
            'message': message,
            'status-code': HTTPStatus.OK,
            'data': {},
        }

    return response
