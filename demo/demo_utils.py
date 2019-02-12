import io
import re
import base64

import numpy as np

from PIL import Image
import cv2

from keras.models import load_model
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing


model_directory = 'model/'

### load model
BACKBONE = "resnet34"
MODEL_FILE = model_directory + 'resnet34_n2048_batch32.h5'
model = Unet(backbone_name=BACKBONE, encoder_weights='imagenet')
model.load_weights(MODEL_FILE)
model._make_predict_function()


def generate_mask(input_img, model = model):
    '''
    Params:
        input_img: PIL jpg image of size [224,224,3]
    Reutrn: 
        ypred: np.array of size [224,224,3] values{0,1}
    '''
    ## process x
    x= np.array(input_img).reshape((1,224,224,3))
    preprocessing_fn = get_preprocessing(BACKBONE)
    x = preprocessing_fn(x)
    ## predict y
    ypred = model.predict(x).reshape((224,224,1))
    ypred = np.dstack((ypred, ypred, ypred))
    return ypred

def mask_overlay(image, mask, alpha = 0.3, color=(0, 1, 0)):
    """
    Helper function to visualize mask on the top of the image
    Params:    
        image: Image object size(244,244,3)
        mask: Image object size(244,244,3) values{0,255}
        color: the color of the mask (choose RGB channel)
        img_format: "image" from .jpg or "array" from image.img_to_array
    Return:
        overlayed image (np.array)
    """ 
    

    mask = (mask * np.array(color)).astype(np.uint8)
    image = (image * np.array(1)).astype(np.uint8)
    
    weighted_sum = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0.)
    img = image.copy()
    RGB_index = sum(color * np.array((0,1,2))) # indicator of which RGB channel R:0, G:1, B:2
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def generate_overlay(input_img, model = model):
    '''
    Params:
        input_img: PIL jpg image of size [224,224,3]
    Return: 
        ypred: np.array, size [224,224,3], values{probability between 0 and 1}
        mask: PIL jpg images, size [224,224,3]
        merge: PIL jpg images, size [224,224,3]
    '''
    ypred = generate_mask(input_img)
    mask  = Image.fromarray((ypred * 255).astype(np.uint8))

    merge = mask_overlay(input_img, mask)
    merge = Image.fromarray((merge).astype(np.uint8))

    return [mask, merge]

def mask_size(mask, cutoff = 0.5):
    '''
    Params:
        mask: np.array, size [224,224,3], values 0-1
    Return:
        size(float): relative size of gb compared to the whole image
    '''
    mask_binary = (mask > cutoff)
    size = np.sum(mask_binary[:,:,0], axis = (0,1)) / (mask_binary.shape[0] * mask_binary.shape[1])
    return size

def image_rgb(image, mask, cutoff = 0.5):
    '''
    Params:
        image: np.array object, size [224,224,3], values 0-255
         mask: np.array object, size [224,224,3], values 0-1
    Return:
        a dictional of pixel values (as a vector) for r,g,b channel
    '''
    df_rgb = {}
    seg = np.multiply(image, mask > cutoff)
    channel = ["r", "g", "b"]

    for i in range(len(channel)):
        df = seg[:,:,i].flatten()
        df_rgb[channel[i]] = df[df > 0]

    return df_rgb

def base64_to_image(contents):
    '''
    Params:
        contents: the <contents> attribute of dcc.upload component
    Return:
        PIL image object
    '''
    msg = re.sub('^data:image/.+;base64,', '', contents)
    msg = base64.b64decode(msg)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img

def image_to_base64(image):
    '''
    Params:
        image: PIL image object
    Return:
        base64_str: str encoding
    '''
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode()
    return base64_str

