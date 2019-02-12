import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event

import matplotlib.pyplot as plt
import scipy
import numpy as np

import glob
import os
import flask

from io import StringIO
from PIL import Image
import base64
import cv2

from keras.models import load_model
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing





external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

image_directory = '/Users/cc/Desktop/Insight/demo_dash/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory))]
static_image_route = '/static/'


## load model
BACKBONE = "resnet34"
MODEL_FILE = image_directory + 'resnet34_n2048_batch32.h5'
model = Unet(backbone_name=BACKBONE, encoder_weights='imagenet')
model.load_weights(MODEL_FILE)
model._make_predict_function()


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[

    html.Div(style={'width': '100%', 'height': 10}), # space

    html.H2(children='Smart Camera for Laparoscopic Surgery'),

    html.Div(style={'width': '100%', 'height': 5}), # space
    
    ################### demo1 ####################
    #### header
    html.H3(children='Demo: Segmentation of the Gallbladder'),

    #### space
    html.Div(style={'width': '100%', 'height': 10}),

    dcc.Dropdown(
        id='train_image',
        options=[{'label': i, 'value': i} for i in list_of_images],
        value = None,
        placeholder="Select a image for segmentation",
        style = {'width': '50%', 
                 'margin-left': 20}
    ),


    html.Div(style={'width': '100%', 'height': 10}), # space

    #### Box 1
    html.Div([
        html.Label('Image', style = {'fontSize': 20}),
        html.Img(id = 'input_image1', 
                 style = {'width' : 300 , 'height': 300,
                          'outline': 'solid'}),
        html.Div(style={'height': 10}),
        ], style = {'width': '30%', 'display': 'inline-block',
                    'float': 'left', 'margin-left': 50,
                    'vertical-align': 'middle'}
    ),
    

    #### Box 2
    html.Div([
        html.Label('Generated Mask', style = {'fontSize': 20}),
        html.Img(id = 'mask1', 
                 style = {'width' : 300 , 'height': 300,
                          'outline': 'solid'}),
        html.Div(style={'height': 20}),
        html.Button('Segmentation', id = 'segment1', type = "submit"),
        ], style = {'width': '33%', 'display': 'inline-block',
                    'vertical-align': 'middle'}
    ),

    #### Box 3
    html.Div([
        html.Label('Overlay', style = {'fontSize': 20}),
        html.Img(id = 'merge1',
                 style = {'width' : 300 , 'height': 300,
                          'outline': 'solid'}),
        ], style = {'width': '33%', 'display': 'inline-block',
                    'float': 'right', 'margin-right': 5,
                    'vertical-align': 'middle'}
    ),



    #############################################################

    html.Div(style={'width': '100%', 'height': 50}), # space

    
    dcc.Upload(id = 'upload_image',
               children = html.Button('Upload Your Own Image'),
               style = {'width': '50%', 
                        'margin-left': 40}),

    html.Div(style={'width': '100%', 'height': 10}), # space

    #### Box 4
    html.Div([
        html.Label('Image', style = {'fontSize': 20}),
        html.Img(id = 'input_image2',
                 style = {'width' : 300 , 'height': 300,
                          'outline': 'solid'}),
        html.Div(style={'height': 10})
        ], style = {'width': '30%', 'display': 'inline-block',
                    'float': 'left', 'margin-left': 50,
                    'vertical-align': 'middle'}
    ),
    

    #### Box 5
    html.Div([
        html.Label('Generated Mask', style = {'fontSize': 20}),
        html.Img(id = 'mask2',
                 style = {'width' : 300 , 'height': 300,
                          'outline': 'solid'}),
        html.Div(style={'height': 20}),
        html.Button('Segmentation', id = 'segment2'),
        ], style = {'width': '33%', 'display': 'inline-block',
                    'vertical-align': 'middle'}
    ),

    #### Box 6
    html.Div([
        html.Label('Overlay', style = {'fontSize': 20}),
        html.Img(id = 'merge2',
                 style = {'width' : 300 , 'height': 300,
                          'outline': 'solid'}),
        ], style = {'width': '33%', 'display': 'inline-block',
                    'float': 'right', 'margin-right': 5,
                    'vertical-align': 'middle'}
    )


    ]
)



def generate_mask(input_img, model = model):
    '''
    input: PIL jpg image of size [224,224,3]
    output: np.array of size [224,224,3]
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
    input: PIL jpg image of size [224,224,3]
    output: PIL jpg imagesize [224,224,3]
    '''
    ypred = generate_mask(input_img)
    ypred  = Image.fromarray((ypred * 255).astype(np.uint8))

    merge = mask_overlay(input_img, ypred)
    merge = Image.fromarray((merge).astype(np.uint8))

    return [ypred, merge]


@app.callback(Output('input_image1', 'src'),
             [Input('train_image', 'value')])
def choose_image(value):
    if value != None:
        input_img = Image.open(image_directory + value) # PIL jpg object
        input_img = input_img.resize((224,224))
        mask, merge = generate_overlay(input_img)

        mask.save("mask" + value[-5] + ".png")
        merge.save("merge" + value[-5] + ".png")

        return (static_image_route + value)



@app.callback(Output('input_image2', 'src'),
              [Input('upload_image', 'contents')])
def upload_image(contents):
    return contents


@app.callback(Output('mask1', 'src'),
              [Input('segment1', 'n_clicks')],
              [State('train_image', 'value')])
def update_mask(n_clicks, value):
    if n_clicks != None:
        return (static_image_route + "mask" + value[-5] + ".png")


@app.callback(Output('merge1', 'src'),
              [Input('segment1', 'n_clicks')],
              [State('train_image', 'value')])
def update_merge(n_clicks, value):
    if n_clicks != None:
        return (static_image_route + "merge" + value[-5] + ".png")

'''
@app.callback(Output('mask2', 'src'),
              [Input('segment2', 'n_clicks')],
              [State('input_image2', 'value')])
def update_mask(n_clicks, value):
    if n_clicks != None:
        return (static_image_route + "mask" + value[-5] + ".png")


@app.callback(Output('merge2', 'src'),
              [Input('segment2', 'n_clicks')],
              [State('train_image', 'value')])
def update_merge(n_clicks, value):
    if n_clicks != None:
        return (static_image_route + "merge" + value[-5] + ".png")
'''



@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    return flask.send_from_directory(image_directory, image_name)


if __name__ == '__main__':
    app.run_server(debug=True)