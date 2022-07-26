import gradio as gr
import PIL
import requests
import numpy as np
import cv2

from typing import List
from deployment.utils import encode, decode

def draw_bboxes(image : np.ndarray, boxes : List[float], labels : List[int]):
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

    for i, box in enumerate(boxes):
        # Get the object class color
        color = colors[labels[i]]

        # Get the box coordinates
        [x1, y1, x2, y2] = np.array(box).astype(int)
        # Need to copy to avoid an error in cv2.rectangle
        image = np.array(image).copy()

        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv2.rectangle(image, pt1, pt2, color, thickness=3)
        cv2.putText(image, texts[labels[i]], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, thickness=3, color=color)

    return image

def waste_detector_interface(
    image : PIL.Image,
    detection_threshold : float,
    nms_threshold : float
):
    # Encode the image so it can be sended as a JSON
    encoded_image = encode(image)

    # Body of the request
    payload = {
        "image": encoded_image,
        "nms_threshold": nms_threshold,
        "detection_threshold": detection_threshold
    }

    response = requests.post(url='http://backend:5000/predict', json=payload)

    # Handle errors during inference
    if response.status_code == 500:
        detail = response.json()['detail']
        raise ValueError(detail)

    # Parse the JSON body to obtain a dictionary
    response = response.json()

    # Draw the returned bounding boxes into the original image
    final_image = draw_bboxes(image, response['boxes'], response['labels'])
    final_image = PIL.Image.fromarray(final_image)

    return final_image

def main():
    inputs = [
        gr.components.Image(type="pil", label="Original Image"),
        gr.components.Number(default=0.5, label="detection_threshold"),
        gr.components.Number(default=0.5, label="nms_threshold"),
    ]

    outputs = [
        gr.components.Image(type="pil", label="Prediction"),
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
