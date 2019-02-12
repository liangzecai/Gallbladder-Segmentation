import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.graph_objs as go
import flask

import glob
import os

import numpy as np
import pandas as pd

from demo_utils import *


############################ styles ##############################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

### style
tab_default_style = {'fontSize': 20}
tab_selected_style = {'fontSize': 20, 'backgroundColor': '#86caf9'}

dropdown_style = {'fontSize': 18}

button_style = {'fontSize': 14, 'textAlign': 'center'}

label_style = {'fontSize': 20, 'fontWeight': 'bold'}
LEFT_MARGIN = 40

########################## image files ##############################
### set up static folder
image_directory = 'img/'
# images for demo1
list_of_images1 = [os.path.basename(x) for x in glob.glob('{}new*.jpg'.format(image_directory))]
# images for demo2
list_of_images2 = [os.path.basename(x) for x in glob.glob('{}fun*.jpg'.format(image_directory))]
static_image_route = '/static/'

############################ data ##############################
df_size = pd.read_csv('data/df_mask_size_distribution.csv', names = ['size'])
MASK_CUTOFF = 0.5

slide_url = 'https://docs.google.com/presentation/d/e/2PACX-1vQvGP8Wp2Qd3pc7l31TZP_rNm6ka3I2QD9krNsq3epRMj9D-hI55huyJFe5b05S820vvRQ4mAiyZr0y/embed?start=false&loop=false&delayms=3000'
###############################################################

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[

    html.Div(style={'width': '100%', 'height': 10}), # space

    html.H2(children='LaparoVision: Organ Segmentation for Smart Camera in Laparoscopic Surgery',
            style = {'margin-left': LEFT_MARGIN}),

    html.Div(style={'width': '100%', 'height': 20}), # space
    
    
    dcc.Tabs(id = 'tabs', children = [
        
############################## Tab 1 #############################################
        #### header
        dcc.Tab(label = 'Gallbladder Segmentation', children = [
                html.Div(style={'width': '100%', 'height': 10}),
                html.H6(children='How does the model perform on test data?',
                        style = {'margin-left': LEFT_MARGIN, 'fontWeight': 'bold'}),
                html.Div(style={'width': '100%', 'height': 10}),
                html.H6(children='Select an image that the model has never seen, and click SEGMENTATION.',
                        style = {'margin-left': LEFT_MARGIN}),

                #### space
                html.Div(style={'width': '100%', 'height': 10}),

                #### dropdown and button
                html.Div(children = [
                    dcc.Dropdown(
                        id='train_image1',
                        options=[{'label': i.split("_")[2], 'value': i} for i in list_of_images1],
                        value = None,
                        placeholder='Select an image for segmentation',
                        style = dropdown_style,
                        className="four columns",
                    ),

                    html.Button('Segmentation', id = 'segment1', type = "submit",
                                style = button_style , className="two columns",),

                ], className='row', style = {'margin-left' : LEFT_MARGIN}),
                #### end of dropdown and button

                
                html.Div(style={'width': '100%', 'height': 20}), # space


                #### three boxes
                html.Div(children = [
                    #### Box 1
                    html.Div([
                        html.Label('Image', style = label_style),
                        html.Img(id = 'input_image1', 
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),
                

                    #### Box 2
                    html.Div([
                        html.Label('Generated Mask', style = label_style),
                        html.Img(id = 'mask1', 
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className="three columns",
                    ),

                    #### Box 3
                    html.Div([
                        html.Label('Overlay', style = label_style),
                        html.Img(id = 'merge1',
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),   
        

                ], className='row', style = {'margin-left': LEFT_MARGIN}),
                #### end of three boxes
           
                
                html.Div(style={'width': '100%', 'height': 40}), # space

                html.Div(children = [
                    html.Label('Diagnosis Plots', 
                               style = {'fontSize': 24, 'fontWeight': 'bold'}),

                    #### plot1: distribution of Gb size
                    html.Div(id = 'plot_size'),

                    #### plot2: RGB histogram
                    html.Div(id = 'plot_rgb'),

                ], style = {'margin-left': LEFT_MARGIN, 'width': '80%'},)
                
                

        ], 
        style = tab_default_style,
        selected_style = tab_selected_style),



################################ Tab 2 ########################################

        dcc.Tab(label = 'Look into the Model (for Fun)', children = [

            html.Div(style={'width': '100%', 'height': 10}), # space
            html.H6(children='What is the gallbladder-ness that the model has learned?',
                        style = {'margin-left': 40, 'fontWeight': 'bold'}),
            html.Div(style={'width': '100%', 'height': 10}),
            html.H6(children='Select one of the fun images, and click SEGMENTATION to see what the model would detect as gallbladder.',
                        style = {'margin-left': 40}),
            html.Div(style={'width': '100%', 'height': 10}), 
            html.H6(children='Pay attention to what the model might be detecting: object color, shape and texture',
                        style = {'margin-left': 40}),
            html.Div(style={'width': '100%', 'height': 10}), 

            html.Div(children = [
                    dcc.Dropdown(
                        id='train_image2',
                        options=[{'label': i.split("_")[2], 'value': i} for i in list_of_images2],
                        value = None,
                        placeholder='Select an image for segmentation',
                        style = dropdown_style,
                        className='four columns',
                    ),

                    html.Button('Segmentation', id = 'segment2', type = "submit",
                                style = button_style , className="two columns",),

            ], className='row', style = {'margin-left' : LEFT_MARGIN}),
        
            

            html.Div(style={'width': '100%', 'height': 20}), # space

            #### three boxes
            html.Div(children = [
                    #### Box 4
                    html.Div([
                        html.Label('Image', style = label_style),
                        html.Img(id = 'input_image2', 
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),
                

                    #### Box 5
                    html.Div([
                        html.Label('Generated Mask', style = label_style),
                        html.Img(id = 'mask2', 
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),

                    #### Box 6
                    html.Div([
                        html.Label('Overlay', style = label_style),
                        html.Img(id = 'merge2',
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),   
         

                ], className='row', style = {'margin-left': LEFT_MARGIN}),


            


        ], 
        style = tab_default_style,
        selected_style = tab_selected_style),



############################# Tab 3 ###########################

        dcc.Tab(label = 'Upload Your Own Image', children = [
            html.Div(style={'width': '100%', 'height': 10}), # space

            
            html.H6(children='Upload your own image, and click SEGMENTATION to see what the model would detect.',
                        style = {'margin-left': 40}),
            html.Div(style={'width': '100%', 'height': 10}), 

            html.Div(children = [
                    dcc.Upload(html.Button('Upload File', style = button_style), 
                               id = 'upload_image',
                               className='two columns'),

                    html.Button('Segmentation', id = 'segment3', type = "submit",
                                style = button_style , className="two columns",),

            ], className='row', style = {'margin-left' : LEFT_MARGIN}),
        
        
            html.Div(style={'width': '100%', 'height': 20}), # space

            #### three boxes
            html.Div(children = [
                    #### Box 7
                    html.Div([
                        html.Label('Image', style = label_style),
                        html.Img(id = 'input_image3', 
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),
                

                    #### Box 8
                    html.Div([
                        html.Label('Generated Mask', style = label_style),
                        html.Img(id = 'mask3', 
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),

                    #### Box 9
                    html.Div([
                        html.Label('Overlay', style = label_style),
                        html.Img(id = 'merge3',
                                 style = {'width' : 300 , 'height': 300,
                                          'outline': 'solid'}),
                        ], className='three columns',
                    ),   
         

                ], className='row', style = {'margin-left': LEFT_MARGIN}),
        ],
        style = tab_default_style,
        selected_style = tab_selected_style),



###################################### Tab 4: slides ###########################
        dcc.Tab(label = 'Read More (Slides)', children = [
            html.Div(style={'width': '100%', 'height': 50}), # space
            html.Iframe(src = slide_url,
                       style={'border': 'none', 'width': 960, 'height': 569, 
                              'textAlign': 'center', 'margin-left': 100})
            ],
            style = tab_default_style,
            selected_style = tab_selected_style)
    
    ])

])



################## callbacks for demo1 #######################

## choose a new_image from dropdown menu,
## generate and save mask.jpg and merge.jpg
## show new_image in box1
@app.callback(Output('input_image1', 'src'),
             [Input('train_image1', 'value')])
def choose_image(value):
    if value != None:
        input_img = Image.open(image_directory + value) # PIL jpg object
        input_img = input_img.resize((224,224))
        mask, merge = generate_overlay(input_img)

        ## save mask and overlay images
        name = value.split('_')[2].split('.')[0]
        mask.save(image_directory  + name + '_mask.jpg')
        merge.save(image_directory  + name + '_merge.jpg')

        return (static_image_route + value)


## choose a new_image from dropdown menu,
## calculate the mask size
## plot on plot_size

@app.callback(Output('plot_size', 'children'),
             [Input('train_image1', 'value')])
def update_plot_size(value):
    # default plot when no image is chosen
    new_size = 0

    if value != None:
        input_img = Image.open(image_directory + value) # PIL jpg object
        input_img = input_img.resize((224,224))
        ypred = generate_mask(input_img) # np.array object
        new_size = mask_size(ypred, cutoff = MASK_CUTOFF)


    p = dcc.Graph(figure = {
                    'data': [{'x': df_size['size'].tolist(),
                            'type': 'histogram',
                            'name': 'Gallbladder Size Distribution (N = 2420)'},
                            {'x': [new_size, new_size],
                            'y': [0, 150],
                            'type': 'line',
                            'name': 'new image'}],
                    'layout': {'height': 400,
                                'title': 'Size of the Segmented Gallbladder',
                                'titlefont': {'size':22},
                                'xaxis':{'title':"Gallbladder Pixels / Total Pixels of the Image"},
                                'yaxis':{'title':'Count'},
                                'legend':{'x': 0.7, 'y': 1},
                                    }
                        }
                ),
    return p

## choose a new_image from dropdown menu,
## plot rgb histogram on plot_rgb
@app.callback(Output('plot_rgb', 'children'),
             [Input('train_image1', 'value')])
def update_plot_rgb(value):
    # default plot when no image is chosen
    df_rgb = {'r': [0,0,0],
              'g': [0,0,0],
              'b': [0,0,0]}

    if value != None:
        input_img = Image.open(image_directory + value) # PIL jpg object
        input_img = input_img.resize((224,224))
        ypred = generate_mask(input_img)
        df_rgb_new = image_rgb(np.array(input_img), ypred)
        df_rgb['r'] = df_rgb_new['r']
        df_rgb['g'] = df_rgb_new['g']
        df_rgb['b'] = df_rgb_new['b']


    p = dcc.Graph(figure = {
                    'data': [{'x': df_rgb['r'],
                            'type': 'histogram', 
                            'marker': {'color': '#DC143C'}, 
                            'name': 'Red Channel', 'opacity': 0.8},
                            {'x': df_rgb['g'],
                            'type': 'histogram', 
                            'marker': {'color': "#008000"},
                            'name': 'Green Channel', 'opacity': 0.8},
                            {'x': df_rgb['b'],
                            'type': 'histogram', 
                            'marker': {'color': '#4169E1'},
                            'name': 'Blue Channel', 'opacity': 0.8},
                            ],
                    'layout': {'height': 400,
                               'title': 'Color Histogram of the Segmented Gallbladder',
                               'titlefont': {'size':22},
                                'xaxis':{'title':"Pixel Value"},
                                'yaxis':{'title':'Count'},
                                'legend':{'x': 0.7, 'y': 1},
                                    }
                        }
                ),

    return p


## when click SEGMENTATION
## show mask.jpg in box2
@app.callback(Output('mask1', 'src'),
              [Input('segment1', 'n_clicks')],
              [State('train_image1', 'value')])
def update_mask(n_clicks, value):
    if n_clicks != None:
        name = value.split('_')[2].split('.')[0]
        return (static_image_route + name + '_mask.jpg')


## when click SEGMENTATION
## show merge.jpg in box3
@app.callback(Output('merge1', 'src'),
              [Input('segment1', 'n_clicks')],
              [State('train_image1', 'value')])
def update_merge(n_clicks, value):
    if n_clicks != None:
        name = value.split('_')[2].split('.')[0]
        return (static_image_route + name + '_merge.jpg')


################## callbacks for demo2 #######################

## choose a fun_image from dropdown menu,
## generate and save mask.jpg and merge.jpg
## show new_image in box4
@app.callback(Output('input_image2', 'src'),
             [Input('train_image2', 'value')])
def choose_image(value):
    if value != None:
        input_img = Image.open(image_directory + value) # PIL jpg object
        input_img = input_img.resize((224,224))
        mask, merge = generate_overlay(input_img)

        name = value.split('_')[2].split('.')[0]
        mask.save(image_directory  + name + '_mask.jpg')
        merge.save(image_directory  + name + '_merge.jpg')

        return (static_image_route + value)


## when click SEGMENTATION
## show mask.jpg in box5
@app.callback(Output('mask2', 'src'),
              [Input('segment2', 'n_clicks')],
              [State('train_image2', 'value')])
def update_mask(n_clicks, value):
    if n_clicks != None:
        name = value.split('_')[2].split('.')[0]
        return (static_image_route + name + '_mask.jpg')

## when click SEGMENTATION
## show merge.jpg in box6
@app.callback(Output('merge2', 'src'),
              [Input('segment2', 'n_clicks')],
              [State('train_image2', 'value')])
def update_merge(n_clicks, value):
    if n_clicks != None:
        name = value.split('_')[2].split('.')[0]
        return (static_image_route + name + '_merge.jpg')



################## callbacks for Tab3: upload image #######################

## upload image
@app.callback(Output('input_image3', 'src'),
              [Input('upload_image', 'contents')])
def upload_image(contents):
    return contents

## when click SEGMENTATION
## show mask.jpg in box8
@app.callback(Output('mask3', 'src'),
              [Input('segment3', 'n_clicks')],
              [State('upload_image', 'contents')])
def update_mask(n_clicks, contents):
    if contents != None:
        input_img = base64_to_image(contents) # PIL object
        input_img = input_img.resize((224,224))
        mask, merge = generate_overlay(input_img)
        return 'data:image/png;base64,{}'.format(image_to_base64(mask))


## when click SEGMENTATION
## show mask.jpg in box9
@app.callback(Output('merge3', 'src'),
              [Input('segment3', 'n_clicks')],
              [State('upload_image', 'contents')])
def update_mask(n_clicks, contents):
    if contents != None:
        input_img = base64_to_image(contents) # PIL object
        input_img = input_img.resize((224,224))
        mask, merge = generate_overlay(input_img)
        return 'data:image/png;base64,{}'.format(image_to_base64(merge))



################## general #######################

## access files from static folder
@app.server.route('{}<image_path>.jpg'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.jpg'.format(image_path)
    return flask.send_from_directory(image_directory , image_name)


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port = 5000)