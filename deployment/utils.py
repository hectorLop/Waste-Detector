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

def get_data_drift(image : PIL.Image, data_dist : Dict):
    """
    Measure the data drift between a given image and the
    training data distribution.

    Args:
        image (PIL.Image): Image uploaded for inference.
        data_dist (Dict): Dictionary containing the training data
            distributions for the hue, brightness and saturation.

    Returns:
        tuple: A tuple containing:
            (float): Distance between the training data hue distribution
                and the inference image hue distribution.
            (float): Distance between the training data saturation 
                distribution and the inference image saturation distribution.
            (float): Distance between the training data brightness 
                distribution and the inference image brightness distribution.
    """
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = image[:, :, 0]
    hue_pdf, _ = np.histogram(hue, bins=np.arange(0, 255, 10), density=True)

    saturation = image[:, :, 1]
    saturation_pdf, _ = np.histogram(saturation, bins=np.arange(0, 255, 10), density=True)

    brightness = image[:, :, 2]
    brightness_pdf, _ = np.histogram(brightness, bins=np.arange(0, 255, 10), density=True)

    hue_dist = jensenshannon(data_dist['hue'], hue_pdf)
    saturation_dist = jensenshannon(data_dist['saturation'], saturation_pdf)
    brightness_dist = jensenshannon(data_dist['brightness'], brightness_pdf)

    return hue_dist, saturation_dist, brightness_dist
