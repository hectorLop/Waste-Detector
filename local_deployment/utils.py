from typing import Tuple, Dict
from scipy.spatial.distance import jensenshannon

import cv2
import numpy as np
import io
import base64
import PIL


def encode(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    image = buf.getvalue()
    image = base64.b64encode(image).decode('utf8')

    return image

def decode(encoded_image):
    img_bytes = base64.b64decode(encoded_image.encode('utf-8'))
    image = PIL.Image.open(io.BytesIO(img_bytes))

    return image
