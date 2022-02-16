from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from datetime import datetime
from http import HTTPStatus
from functools import wraps
import io
import base64
import PIL

from deployment.utils import get_models, encode, decode
from deployment.classifier import CustomEfficientNet, CustomViT
from deployment.model import predict_boxes, prepare_prediction, predict_class


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
    print('Models loaded successfully')

@app.route('/predict', methods=['POST'])
async def predict(request):
    request = await request.json()
    im_b64 = request['image']
    image = decode(im_b64)
    #img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    #image = PIL.Image.open(io.BytesIO(img_bytes))
    detection_threshold = float(request['detection_threshold'])
    nms_threshold = float(request['nms_threshold'])
    
    print('Predicting bounding boxes')
    pred_dict = predict_boxes(detector, image, detection_threshold)
    
    print('Fixing the preds')
    boxes, image = prepare_prediction(pred_dict, nms_threshold)    

    print('Predicting classes')
    labels = predict_class(classifier, image, boxes)

    image = PIL.Image.fromarray(image)
    image = encode(image)

    #buf = io.BytesIO()
    #image.save(buf, format='PNG')
    #image = buf.getvalue()
    #image = base64.b64encode(image).decode('utf8')
 
    payload = {
        'image': image,
        'boxes': boxes.tolist(),
        'labels': labels.tolist()
    }

    return JSONResponse(content=payload)
