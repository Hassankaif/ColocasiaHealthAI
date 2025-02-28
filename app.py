import streamlit as st
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import urllib.request
from PIL import Image
import os
from openai import OpenAI


# Load your model (Make sure you save and load it correctly)
model = load_model('backend/resnet50_final_model.keras')

# Define image size
IMG_SIZE = (224, 224)

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Load class names (Make sure to update them if needed)
class_names = ['Disease_Leaf_Blight_Dorsal', 'Disease_Leaf_Blight_Ventral', 'Disease_Mosaic_Dorsal', 'Disease_Mosaic_Ventral', 'Healthy_Dorsal', 'Healthy_Ventral']

# Streamlit User Interface
st.title("Colocasia Esculenta Leaf Disease Detection")
st.write("Upload an image from your local system or provide a URL of an image.")

# Select input type (Local file or URL)
input_type = st.radio("Select input type", ('Local file', 'Image URL'))

# Upload Image or provide URL
if input_type == 'Local file':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess and predict
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Prediction Confidence: {np.max(prediction) * 100:.2f}%")

elif input_type == 'Image URL':
    img_url = st.text_input("Enter image URL:")
    if img_url:
        # Download the image from URL
        img_path = "temp_image.jpg"
        urllib.request.urlretrieve(img_url, img_path)

        img = Image.open(img_path)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess and predict
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Prediction Confidence: {np.max(prediction) * 100:.2f}%")

        # Clean up the temporary image
        os.remove(img_path)
