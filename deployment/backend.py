from fastapi import FastAPI, Request
from datetime import datetime
from http import HTTPStatus
from functools import wraps

from utils import get_models
from classifier import CustomEfficientNet, CustomViT
from model import get_model, predict, prepare_prediction, predict_class


app = FastAPI(
    title='Waste Detector',
    description='Detect waste in images',
    version='0.1'
)

def construct_response(func):
    """
    Construct a JSON response for an endpoint's results
    """

    @wraps(func)
    def wrap(request : Request, *args, **kwargs):
        results = func(request, *args, **kwargs)

        response = {
            'message': results['message'],
            'method': request.method,
            'status-code': results['status-code'],
            'timestamp': datetime.now().isoformat(),
            'url': request.url._url,
        }

        if 'data' in results:
            response['data'] = results['data']

        return response

    return wrap


@app.get('/', tags=['General'])
@construct_response
def _index(request : Request):
    """
    Health check
    """
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {}
    }

    return response

@app.on_event('startup')
def load_artifacts():
    global detector, classifier
    detector, classifier = get_models()

@app.get('/predict', tags=['Prediction'])
def predict():
    print('Predicting bounding boxes')
    pred_dict = predict(detector, image, detection_threshold)
    print('Fixing the preds')
    boxes, image = prepare_prediction(pred_dict, nms_threshold)

    print('Predicting classes')
    labels = predict_class(classifier, image, boxes)
