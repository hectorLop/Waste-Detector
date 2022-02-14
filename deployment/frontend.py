import gradio as gr
from gradio.networking import get_first_available_port
import PIL
import torch
import os

from utils import plot_img_no_mask, get_models
from classifier import CustomEfficientNet, CustomViT
from model import get_model, predict, prepare_prediction, predict_class

DET_CKPT = 'efficientDet_icevision.ckpt'
CLASS_CKPT = 'class_ViT_taco_7_class.pth'

def waste_detector_interface(
    image,
    detection_threshold,
    nms_threshold
): 
    # TODO: This function send a request to the backend and
    #       receive a response
    pass

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
    ['example_imgs/basura_4_2.jpg', 0.5, 0.5],
    ['example_imgs/basura_1.jpg', 0.5, 0.5],
    ['example_imgs/basura_3.jpg', 0.5, 0.5]
]

gr.close_all()
#port = get_first_available_port(7682, 9000)

gr.Interface(
    waste_detector_interface,
    inputs,
    outputs,
    title=title,
    description=description,
    examples=examples,
    theme="huggingface"
).launch(share=True)

os.system('python3 app.py')
