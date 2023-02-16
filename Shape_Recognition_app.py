#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd

#image processing imports
import tensorflow as tf
#for converting image into array
from tensorflow.keras.preprocessing import image

#imports for model
import keras
from keras.models import load_model
from tensorflow.keras.models import Sequential
from keras.applications.inception_v3 import preprocess_input

# for displaying images when predicting class
from PIL import Image 

lock = threading.Lock()
img_container = {"img": None}

#function to predict image
def predict(uploaded_file, model): 
    #loading image from the path of image
    im = Image.open(uploaded_file)
    # resizing images to all be the same size    
    im = im.resize((224, 224))    
    # makes sure the image is in RGB (converts all images to have only 3 color channels, png images have 4 color channels)
    im = im.convert(mode='RGB')
    # converts image into an array
    im = tf.keras.preprocessing.image.img_to_array(im)
    # expands array (from converted image) with a new dimension (for calculated category)
    im = np.expand_dims(im, axis = 0)
    im = tf.keras.applications.inception_v3.preprocess_input(im)
    #running image through the model to predict class
    prediction = model.predict(im)
    st.write("\n")
    #showing prediction chart with percentage of each class
    st.write(pd.DataFrame(prediction, columns = ['Circle','Square','Triangle'], index = ['Probability']))
    st.write("\n")
    # computing category weither a shape is a rectangle, square, star or triangle
    if prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
        st.write(f'Shape is predicted as a Circle with {"%.2f" % ((prediction[0][0])*100)}% certainty')
    elif prediction[0][1] > prediction[0][0] and prediction[0][1] >  prediction[0][2]:
        st.write(f'Shape is predicted as a Square with {"%.2f" % ((prediction[0][1])*100)}% certainty')
    elif prediction[0][2] > prediction[0][0] and prediction[0][2] > prediction[0][1]:
        st.write(f'Shape is predicted as a Triangle with {"%.2f" % ((prediction[0][2])*100)}% certainty')



st.title('SMART SHAPE RECOGNITION')
st.text('Using Streamlit v1.16.0')
# making Sidebar for navigation and options
activities = ['Mission Statement','Model0', 'Model1','Model2']
choice = st.sidebar.selectbox('SideBar Navigation', activities)
if choice == 'Mission Statement':
    img = Image.open('.models/img2.png')
    st.image([img],width=150)
    st.subheader('Mission Statement')
    st.text('The project attempts to develop an empirical method for evaluating human-drawn shapes. \n')
    st.text('Accuracy, loss validation, and accuracy validation values are differentiated with\n')
    st.text('the CNN model using the Keras library. \n')
    st.text('')


elif choice == 'Model0':        
    st.text('Draw triangle, square or circle and try it!')  
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle")
    )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=448,
    width=448,
    drawing_mode=drawing_mode,
    key="canvas",
    )   

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        #im=st.image(canvas_result.image_data)
        cv2.imwrite(f"img.jpg",  canvas_result.image_data)

    #loading trained model for predicting
    model0 = tf.keras.models.load_model('./models/model_v2.h5')
    #setup for drag and drop to run prediction model
    st.subheader('Shape Recognition & Accuracy Checking')
    uploaded_file = st.file_uploader('Choose an image of a shape to classify...', type= ['png','jpg','jpeg'])
    if uploaded_file is not None:
            #load image file
            im = Image.open(uploaded_file)        
            #display image and caption
            st.image(im, caption='Uploaded Image.',width=200, use_column_width=False)
            st.write("")
            st.text('Using perfect shapes - created dataset 3 CNN layers')
            st.write("Classifying at:")
            st.write("\n")
            #run inputted image through model
            predict(uploaded_file, model0)    
elif choice == 'Model1':
    st.text('Draw triangle, square or circle and try it!')                        
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle")
    )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=448,
    width=448,
    drawing_mode=drawing_mode,
    #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
    )   

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        #im=st.image(canvas_result.image_data)
        cv2.imwrite(f"img.jpg",  canvas_result.image_data)
   
    #loading trained model for predicting
    model1 = tf.keras.models.load_model('./models/model_1.h5')
    #setup for drag and drop to run prediction model
    st.subheader('Shape Recognition & Accuracy Checking')
    uploaded_file = st.file_uploader('Choose an image of a shape to classify...', type= ['png','jpg','jpeg'])
    if uploaded_file is not None:
            #load image file
            im = Image.open(uploaded_file)        
            #display image and caption
            st.image(im, caption='Uploaded Image.',width=200, use_column_width=False)
            st.write("")
            st.text('Using perfect shapes - created dataset 5 CNN layers')
            st.write("Classifying at:")
            st.write("\n")
            #run inputted image through model
            predict(uploaded_file, model1)  

elif choice == 'Model2': 
    st.text('Draw triangle, square or circle and try it!')                        
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle")
    )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=448,
    width=448,
    drawing_mode=drawing_mode,
    #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
    )   

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        #im=st.image(canvas_result.image_data)
        cv2.imwrite(f"img.jpg",  canvas_result.image_data)
    
    #loading trained model for predicting
    model2 = tf.keras.models.load_model('./models/model_2.h5')
    #setup for drag and drop to run prediction model
    st.subheader('Shape Recognition & Accuracy Checking')
    uploaded_file = st.file_uploader('Choose an image of a shape to classify...', type= ['png','jpg','jpeg'])
    if uploaded_file is not None:
            #load image file
            im = Image.open(uploaded_file)        
            #display image and caption
            st.image(im, caption='Uploaded Image.',width=200, use_column_width=False)
            st.write("")
            st.text('Using perfect and hand drawn shapes - created dataset 5 CNN layers')
            st.write("Classifying at:")
            st.write("\n")
            #run inputted image through model
            predict(uploaded_file, model2)

                 
