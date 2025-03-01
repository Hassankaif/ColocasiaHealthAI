import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import urllib.request
from PIL import Image
import os
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load models
resnet_model = load_model('Models/resnet50_final_model.keras')
efficientnet_model = load_model('Models/efficientnetb3_model.keras')

# Define image size
IMG_SIZE = (224, 224)

# Load disease dataset
df = pd.read_csv('colocasia_disease_dataset.csv')

# Manually load FLAN-T5 model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
reasoning_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to generate recommendations
def generate_recommendation(disease, symptoms):
    dataset_entry = df[df['disease'].str.contains(disease, case=False, na=False)]
    if not dataset_entry.empty:
        dataset_remedy = dataset_entry['ai_generated_remedy'].values[0]
        prompt = f"Colocasia {disease} causes {symptoms}. Given this remedy: {dataset_remedy}, refine and provide best practices."
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = reasoning_model.generate(**inputs, max_length=100, min_length=50, do_sample=False)
        refined_remedy = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if len(refined_remedy.split()) > 80:
            refined_remedy = summarizer(refined_remedy, max_length=80, min_length=50, do_sample=False)[0]['summary_text']
        
        return f"🔬 **Refined Recommendation**:\n{refined_remedy}"
    else:
        prompt = f"Suggest a remedy for Colocasia {disease}. Symptoms: {symptoms}. Include treatment, prevention, and best practices."
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = reasoning_model.generate(**inputs, max_length=100, min_length=50, do_sample=False)
        generated_remedy = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return f"🆕 **AI-Generated Recommendation**:\n{generated_remedy}"

# Load class names
class_names = ['Disease_Leaf_Blight_Dorsal', 'Disease_Leaf_Blight_Ventral',
               'Disease_Mosaic_Dorsal', 'Disease_Mosaic_Ventral',
               'Healthy_Dorsal', 'Healthy_Ventral']

# Streamlit UI Enhancements
st.set_page_config(page_title="Leaf Disease Detection", page_icon="🌿", layout="centered")
st.title("🌱 Colocasia Esculenta Leaf Disease Detection")
st.markdown("Upload an image from your local system or provide an image URL to classify leaf diseases.")

# Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose a model:", ('ResNet50', 'EfficientNetB3'))
model = resnet_model if model_choice == 'ResNet50' else efficientnet_model

# Select input type
st.sidebar.header("Image Input")
input_type = st.sidebar.radio("Select input type", ('Local file', 'Image URL'))

st.markdown("### Upload Image")
if input_type == 'Local file':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Upload an image of a leaf for classification.")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown("### Classification Result")
        st.write("Classifying...")

        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Prediction Confidence:** {np.max(prediction) * 100:.2f}%")

        # Generate and display recommendation
        if "Disease" in predicted_class:
            disease_name = predicted_class.split("_")[1] + " " + predicted_class.split("_")[2]
            symptoms = "Detected from classification"
            remedy = generate_recommendation(disease_name, symptoms)
            st.markdown("### Suggested Remedy")
            st.write(remedy)

elif input_type == 'Image URL':
    img_url = st.text_input("Enter image URL:", placeholder="Paste image URL here")
    if img_url:
        img_path = "temp_image.jpg"
        urllib.request.urlretrieve(img_url, img_path)
        
        img = Image.open(img_path)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown("### Classification Result")
        st.write("Classifying...")

        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Prediction Confidence:** {np.max(prediction) * 100:.2f}%")
        
        # Generate and display recommendation
        if "Disease" in predicted_class:
            disease_name = predicted_class.split("_")[1] + " " + predicted_class.split("_")[2]
            symptoms = "Detected from classification"
            remedy = generate_recommendation(disease_name, symptoms)
            st.markdown("### Suggested Remedy")
            st.write(remedy)
        
        os.remove(img_path)
