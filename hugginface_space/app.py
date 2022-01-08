import streamlit as st
import PIL
import torch

from utils import plot_img_no_mask, get_models
from model import predict, prepare_prediction, predict_class

DET_CKPT = 'efficientDet_icevision.ckpt'
CLASS_CKPT = 'class_ViT_taco_7_class.pth'

st.subheader('Upload Custom Image')

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

st.subheader('Example Images')

example_imgs = [
    'example_imgs/basura_4_2.jpg',
    'example_imgs/basura_1.jpg',
    'example_imgs/basura_3.jpg'
]

with st.container() as cont:
    st.image(example_imgs[0], width=150, caption='1')
    if st.button('Select Image', key='Image_1'):
        image_file = example_imgs[0]

with st.container() as cont:
    st.image(example_imgs[1], width=150, caption='2')
    if st.button('Select Image', key='Image_2'):
        image_file = example_imgs[1]

with st.container() as cont:
    st.image(example_imgs[2], width=150, caption='2')
    if st.button('Select Image', key='Image_3'):
        image_file = example_imgs[2]

st.subheader('Detection parameters')

detection_threshold = st.slider('Detection threshold',
                                min_value=0.0,
                                max_value=1.0,
                                value=0.5,
                                step=0.1)

nms_threshold = st.slider('NMS threshold',
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        step=0.1)

st.subheader('Prediction')

if image_file is not None:
    det_model, classifier = get_models(DET_CKPT, CLASS_CKPT)
    
    print('Getting predictions')
    if isinstance(image_file, str):
        data = image_file
    else:
        data = image_file.read()
    pred_dict = predict(det_model, data, detection_threshold)
    print('Fixing the preds')
    boxes, image = prepare_prediction(pred_dict, nms_threshold)

    print('Predicting classes')
    labels = predict_class(classifier, image, boxes)
    print('Plotting')
    plot_img_no_mask(image, boxes, labels)

    img = PIL.Image.open('img.png')
    st.image(img,width=750)