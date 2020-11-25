import streamlit as st
from streamlit_drawable_canvas import st_canvas
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np


# Hide menu hamburger
hide_streamlit_style = """            
                       <style>            
                       #MainMenu {visibility: hidden;}            
                       footer {visibility: hidden;}            
                       </style>            
                       """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Page Layout
st.title("Handwritten Drawable Digit classifier")

# Creating a canvas for drawing an image of digit
# We'll later use this canvas and feed it to a CNN


canvas_result = st_canvas(stroke_width = 20,
                          stroke_color = "white",
                          background_color="black",
                          height = 280,
                          width = 280,
                          drawing_mode = "freedraw",
                          key = "canvas")

predict = st.button("Predict")


model = load_model('model.h5')

def prepare_image(image):
    img = cv2.resize(image.astype('float32'),(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_pediction(image):
    img = prepare_image(image)
    prediction = model.predict(img.reshape(1,28,28,1))

    return np.argmax(prediction[0])

if canvas_result.image_data is not None and predict:
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i+1)
        time.sleep(0.01)

    prediction = get_pediction(canvas_result.image_data)
    st.text("Prediction: {}".format(prediction))
