import gradio as gr
import PIL
import os
import io
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import boto3
import base64

from typing import List
from deployment.utils import encode, decode

def plot_img_no_mask(image : np.ndarray, boxes : List[float], labels : List[int]):
    # Define colors for each class
    colors = {
        0: (255,255,0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0,128,0),
        4: (255,165,0),
        5: (230,230,250),
        6: (192,192,192)
    }
    # Class definitions
    texts = {
        0: 'plastic',
        1: 'dangerous',
        2: 'carton',
        3: 'glass',
        4: 'organic',
        5: 'rest',
        6: 'other'
    }

    boxes = np.array(boxes)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, box in enumerate(boxes):
        # Get the object class color
        color = colors[labels[i]]

        # Get the box coordinates
        [x1, y1, x2, y2] = np.array(box).astype(int)
        # Need to copy to avoid an error in cv2.rectangle
        image = np.array(image).copy()

        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv2.rectangle(image, pt1, pt2, color, thickness=5)
        cv2.putText(image, texts[labels[i]], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, thickness=5, color=color)

    plt.axis('off')
    ax.imshow(image)

    return fig

def waste_detector_interface(
    image,
    detection_threshold,
    nms_threshold
):
    # Resize the image to avoid surprass the Lambda upload limit
    new_h, new_w = 512, 512
    image = image.resize((new_h, new_w))

    fd = io.BytesIO()
    image.save(fd, format='PNG')
    image = fd.getvalue()
    image = base64.b64encode(fd.getvalue())

    # Header definition
    headers = {
        "Content-type": "application/json",
        "Accept": "text/plain"
    }

    # Request body
    payload = {
        "image": image.decode(),
        "nms_threshold": nms_threshold,
        "detection_threshold": detection_threshold
    }

    lambda_client = boto3.client('lambda')
    response = lambda_client.invoke(FunctionName='waste-detector-prediction',
                                    InvocationType='RequestResponse',
                                    Payload=json.dumps(payload))

    # Read and decode the response Payload
    response = json.loads(response['Payload'].read().decode())
    # Get the body
    response = json.loads(response['body'])
    # Decode the image
    image = decode(response['image'])

    return plot_img_no_mask(image, response['boxes'], response['labels'])

def main():
    inputs = [
        gr.inputs.Image(type="pil", label="Original Image"),
        gr.inputs.Number(default=0.5, label="detection_threshold"),
        gr.inputs.Number(default=0.5, label="nms_threshold"),
    ]

    outputs = [
        gr.outputs.Image(type="plot", label="Prediction"),
    ]

    title = 'Waste Detection'
    description = 'Demo for waste object detection. It detects and classify waste in images according to which rubbish bin the waste should be thrown. Upload an image or click an image to use.'
    examples = [
        ['deployment/example_imgs/basura_4_2.jpg', 0.5, 0.5],
        ['deployment/example_imgs/basura_1.jpg', 0.5, 0.5],
        ['deployment/example_imgs/basura_3.jpg', 0.5, 0.5]
    ]

    gr.Interface(
        waste_detector_interface,
        inputs,
        outputs,
        title=title,
        description=description,
        examples=examples,
        theme="huggingface"
    ).launch(server_port=8501, server_name="0.0.0.0")

if __name__ == '__main__':
    gr.close_all()
    main()
