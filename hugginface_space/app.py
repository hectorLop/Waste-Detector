import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import torch

from classifier import CustomEfficientNet, CustomViT
from model import get_model, predict, prepare_prediction, predict_class

print('Creating the model')
model = get_model('efficientDet_icevision.ckpt')
print('Loading the classifier')
classifier = CustomViT(target_size=7, pretrained=False)
classifier.load_state_dict(torch.load('class_ViT_taco_7_class.pth', map_location='cpu'))
# Set eval mode to deactivate dropout and BN layers 
classifier.eval()

def plot_img_no_mask(image, boxes, labels):
    colors = {
        0: (255,255,0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0,128,0),
        4: (255,165,0),
        5: (230,230,250),
        6: (192,192,192)
    }

    texts = {
        0: 'plastic',
        1: 'dangerous',
        2: 'carton',
        3: 'glass',
        4: 'organic',
        5: 'rest',
        6: 'other'
    }

    # Show image
    boxes = boxes.cpu().detach().numpy().astype(np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, box in enumerate(boxes):
        color = colors[labels[i]]

        [x1, y1, x2, y2] = np.array(box).astype(int)
        # Si no se hace la copia da error en cv2.rectangle
        image = np.array(image).copy()

        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv2.rectangle(image, pt1, pt2, color, thickness=5)
        cv2.putText(image, texts[labels[i]], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, thickness=5, color=color)


    plt.axis('off')
    ax.imshow(image)
    fig.savefig("img.png", bbox_inches='tight')

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
    print('Getting predictions')
    if isinstance(image_file, str):
        data = image_file
    else:
        data = image_file.read()
    pred_dict = predict(model, data, detection_threshold)
    print('Fixing the preds')
    boxes, image = prepare_prediction(pred_dict, nms_threshold)

    print('Predicting classes')
    labels = predict_class(classifier, image, boxes)
    print('Plotting')
    plot_img_no_mask(image, boxes, labels)

    img = PIL.Image.open('img.png')
    st.image(img,width=750)