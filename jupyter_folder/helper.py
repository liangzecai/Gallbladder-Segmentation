import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import keras.backend as K
import keras.applications as ka



#### load data

def load_image_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img[:,:,::-1], (224,224)) ## convert to rgb
        if img is not None:
            images.append(img)
    return np.array(images)

def sample_image_from_folder(folder, sample_list):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename in sample_list:
            img = cv2.imread(os.path.join(folder,filename))
            img = cv2.resize(img[:,:,::-1], (224,224)) ## convert to rgb
            if img is not None:
                images.append(img)
    return np.array(images)


#### preprocessing

identical = lambda x: x
bgr_transpose = lambda x: x[..., ::-1]

models_preprocessing = {
    'vgg16': ka.vgg16.preprocess_input,
    'vgg19': ka.vgg19.preprocess_input,
    'resnet18': bgr_transpose,
    'resnet34': bgr_transpose,
    'resnet50': bgr_transpose,
    'resnet101': bgr_transpose,
    'resnet152': bgr_transpose,
    'resnext50': identical,
    'resnext101': identical,
    'densenet121': ka.densenet.preprocess_input,
    'densenet169': ka.densenet.preprocess_input,
    'densenet201': ka.densenet.preprocess_input,
    'inceptionv3': ka.inception_v3.preprocess_input,
    'inceptionresnetv2': ka.inception_resnet_v2.preprocess_input,
}


def get_preprocessing(backbone):
    """Returns pre-processing function for image data according to name of backbone
    Args:
        backbone (str): name of classification model
    Returns:
        ``callable``: preprocessing_function
    """
    return models_preprocessing[backbone]



#### plot

def mask_overlay(image, mask, alpha = 0.3, color=(0, 1, 0), img_format = "image"):
    """
    Helper function to visualize mask on the top of the image
    Params:    
        image: np.array size(244,244,3) or Image object size(244,244,3)
        mask: np.array size(244,244) values{0,1} or Image object size(244,244,3) values{0,255}
        color: the color of the mask (choose RGB channel)
        img_format: "image" from .jpg or "array" from image.img_to_array
    Return:
        overlayed image (np.array)
    """ 
    
    if img_format == "image":
        mask = (mask * np.array(color)).astype(np.uint8)
        image = (image * np.array(1)).astype(np.uint8)
    
    weighted_sum = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0.)
    img = image.copy()
    RGB_index = sum(color * np.array((0,1,2))) # indicator of which RGB channel R:0, G:1, B:2
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img


def plot_history(history, key = 'loss'):
    train_list = [s for s in history.history.keys() if key in s and 'val' not in s]
    val_list = [s for s in history.history.keys() if key in s and 'val' in s]

    
    if len(train_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[train_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in train_list:
        plt.plot(epochs, history.history[l], 'b', 
                 label='Training ' + key + '(' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_list:
        plt.plot(epochs, history.history[l], 'g', 
                 label='Validation ' + key + '(' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title(key)
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.legend()
    plt.show()


#### loss functino / metrics


def jaccard(y_true, y_pred):
    smooth = 1e-15
    intersection = K.sum(y_true * y_pred, axis=[-1,1,2])
    union = K.sum(y_true + y_pred, axis=[-1,1,2])
    jac = (intersection + smooth) / (union - intersection + smooth)
    return K.mean(jac)

def jaccardLoss(y_true, y_pred):
    J = jaccard(y_true, y_pred)
    return (-K.log(J))

def dice(y_true, y_pred):
    smooth = 1e-15
    intersection = K.sum(y_true * y_pred, axis=[-1,1,2])
    union = K.sum(y_true + y_pred, axis=[-1,1,2])
    return (2 * intersection + 1e-15) / (union + 1e-15)

def NLL(y_true, y_pred):
    H = K.mean(K.binary_crossentropy(y_true, y_pred), axis = [-1,1,2])
    return K.mean(H)

def customLoss(y_true, y_pred):
    H = NLL(y_true, y_pred)
    J = jaccard(y_true, y_pred)
    return (H-K.log(J))
