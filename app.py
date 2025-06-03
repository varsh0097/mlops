import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import os

# Define paths
ARTIFACTS_DIR = "/app/artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.lungcancercode.h5")
CLASS_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "class_indices.pkl")
model = load_model(MODEL_PATH)



# Load model
model = load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, "rb") as f:
    class_indices = pickle.load(f)
class_names = {v: k for k, v in class_indices.items()}

def preprocess_image(img):
    """Preprocess uploaded image for model prediction"""
    img = image.load_img(img, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Streamlit UI
st.title("Lung Cancer Classification")
st.write("Upload a CT scan image to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    try:
        processed_img = preprocess_image(uploaded_file)
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        st.write(f"**Predicted Class:** {class_names.get(predicted_class, 'Unknown')}")
        st.write(f"**Confidence:** {confidence:.2f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
