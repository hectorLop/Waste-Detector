import PIL
import numpy as np

def crop_img_to_bbox(image, bbox):
    img = image.copy()
    bbox = np.array(bbox).astype(int)
    cropped_img = PIL.Image.fromarray(img).crop(bbox)

    return np.array(cropped_img)