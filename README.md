# 🌿 Colocasia Esculenta Leaf Disease Detection

This project is a Leaf Disease Detection System for Colocasia Esculenta using Deep Learning and NLP-based Recommendations. The system classifies leaf images using ResNet50 and EfficientNetB3 models and provides AI-generated remedies for detected diseases using Hugging Face Transformers.

## 🚀 Features

Dual Model Classification: Choose between ResNet50 and EfficientNetB3.

Image Input Options: Upload an image or provide an image URL.

AI-Powered Recommendations: Uses FLAN-T5 and BART for treatment suggestions.

Streamlit UI: User-friendly interface for easy interaction.

## Frontend
![Slide 1]('https://github.com/Hassankaif/ColocasiaHealthAI/blob/main/Visualization/F1.png')
![Slide 2]('https://github.com/Hassankaif/ColocasiaHealthAI/blob/main/Visualization/F2.png')
![Slide 3]('https://github.com/Hassankaif/ColocasiaHealthAI/blob/main/Visualization/F3.png')

## Training InSights
![Slide 3]('https://github.com/Hassankaif/ColocasiaHealthAI/blob/main/Visualization/performance_radar.png')
![Slide 3]('https://github.com/Hassankaif/ColocasiaHealthAI/blob/main/Visualization/pest_detection_performance.png')
![Slide 3]('https://github.com/Hassankaif/ColocasiaHealthAI/blob/main/Visualization/pest_detection_convergence.png')


## 🛠️ Installation

### 1️⃣ Clone the Repository

git clone https://github.com/your-repo/colocasia-leaf-disease.git
cd colocasia-leaf-disease

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
``` 
python -m venv venv
source venv/bin/activate  
```
#### On Windows use: 
```
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Download Required Models
Ensure the classification models and Hugging Face models are correctly set up.
```
mkdir Models
```
Place 'resnet50_final_model.keras' and 'efficientnetb3_model.keras' inside the Models folder.

Manually download Hugging Face models to avoid tokenizer errors:
```
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").save_pretrained("huggingface_models/flan-t5-large")
AutoTokenizer.from_pretrained("google/flan-t5-large").save_pretrained("huggingface_models/flan-t5-large")
```
## 🎯 Usage

### 1️⃣ Run the Streamlit App

streamlit run app.py

### 2️⃣ Upload an Image

Choose Local File or Image URL.

The model will classify the image and display the disease name.

If a disease is detected, the system provides AI-generated remedies.

## 📂 Project Structure

📦 colocasia-leaf-disease

├── 📂 Models               # Pretrained classification models

├── 📂 huggingface_models   # Saved Hugging Face models

├── 📜 app.py               # Streamlit application

├── 📜 colocasia_disease_dataset.csv  # Disease dataset

├── 📜 requirements.txt     # Required dependencies

└── 📜 README.md            # Project documentation


## 🔧 Dependencies

TensorFlow & Keras (for Image Classification)

Hugging Face Transformers (for NLP-based Recommendations)

Streamlit (for UI)

Pandas & NumPy (for data handling)

PIL & urllib (for image processing)

## 📌 Notes

Make sure to have a stable internet connection when using Hugging Face models.

If you face errors with FLAN-T5, manually download and load it as shown above.

## 🤝 Contributing

Feel free to fork this repository, submit issues, or create pull requests to improve the project.

## 📜 License

This project is licensed under the MIT License.

#### Made with ❤️ by The following Team Members From Rajalakshmi Institute of Technology, Chennai.
1. Hassan (ME)
2. Lavanya
3. Mohammed Muhsin
4. Nishanthini
