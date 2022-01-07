from io import BytesIO
from typing import Union
from icevision import *
import collections
import PIL
import torch
import numpy as np
import torchvision

from classifier import transform_image

import icevision.models.ross.efficientdet

MODEL_TYPE = icevision.models.ross.efficientdet

def get_model(checkpoint_path : str):
    extra_args = {}
    backbone = MODEL_TYPE.backbones.d0
    # The efficientdet model requires an img_size parameter
    extra_args['img_size'] = 512

    model = MODEL_TYPE.model(backbone=backbone(pretrained=True),
                             num_classes=2, 
                             **extra_args)
    
    ckpt = get_checkpoint(checkpoint_path)
    model.load_state_dict(ckpt)

    return model

def get_checkpoint(checkpoint_path : str):
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    fixed_state_dict = collections.OrderedDict()

    for k, v in ckpt['state_dict'].items():
        new_k = k[6:]
        fixed_state_dict[new_k] = v

    return fixed_state_dict

def predict(model : object, image : Union[str, BytesIO], detection_threshold : float):
    img = PIL.Image.open(image)
    #img = PIL.Image.open(BytesIO(image))
    img = np.array(img)
    img = PIL.Image.fromarray(img)
    class_map = ClassMap(classes=['Waste'])
    transforms = tfms.A.Adapter([
                    *tfms.A.resize_and_pad(512),
                    tfms.A.Normalize()
                ])

    pred_dict  = MODEL_TYPE.end2end_detect(img,
                                           transforms, 
                                           model,
                                           class_map=class_map,
                                           detection_threshold=detection_threshold,
                                           return_as_pil_img=False,
                                           return_img=True,
                                           display_bbox=False,
                                           display_score=False,
                                           display_label=False)

    return pred_dict

def prepare_prediction(pred_dict, threshold):
    boxes = [box.to_tensor() for box in pred_dict['detection']['bboxes']]
    boxes = torch.stack(boxes)

    scores = torch.as_tensor(pred_dict['detection']['scores'])
    labels = torch.as_tensor(pred_dict['detection']['label_ids'])
    image = np.array(pred_dict['img'])

    fixed_boxes = torchvision.ops.batched_nms(boxes, scores, labels, threshold)
    boxes = boxes[fixed_boxes, :]

    return boxes, image

def predict_class(classifier, image, bboxes):
    preds = []

    for bbox in bboxes:
        img = image.copy()
        bbox = np.array(bbox).astype(int)
        cropped_img = PIL.Image.fromarray(img).crop(bbox)
        cropped_img = np.array(cropped_img)
        #cropped_img = torch.as_tensor(cropped_img, dtype=torch.float).unsqueeze(0)

        tran_image = transform_image(cropped_img, 224)
        tran_image = tran_image.transpose(2, 0, 1)
        tran_image = torch.as_tensor(tran_image, dtype=torch.float).unsqueeze(0)
        print(tran_image.shape)
        y_preds = classifier(tran_image)
        preds.append(y_preds.softmax(1).detach().numpy())

    preds = np.concatenate(preds).argmax(1)

    return preds