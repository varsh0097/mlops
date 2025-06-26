import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import pickle
import h5py

print("TensorFlow version:", tf.__version__)

# Paths
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.lungcancercode.h5")
CLASS_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "class_indices.pkl")

# Safe loader that avoids batch_shape error
def load_legacy_model(model_path):
    with h5py.File(model_path, 'r') as f:
        config = f.attrs.get('model_config')
        if config is None:
            raise ValueError("No model_config found in HDF5 file.")
        if isinstance(config, bytes):
            config = config.decode('utf-8')
        model = model_from_json(config)
        model.load_weights(model_path)
        return model

# Load model
try:
    model = load_legacy_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load class indices
try:
    with open(CLASS_INDICES_PATH, "rb") as f:
        class_indices = pickle.load(f)
    class_names = {v: k for k, v in class_indices.items()}
except Exception as e:
    st.error(f"❌ Failed to load class index file: {e}")
    st.stop()

# Preprocessing
def preprocess_image(img):
    img = image.load_img(img, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Streamlit App
st.title("🫁 Lung Cancer Classification")
st.write("Upload a CT scan image to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 Classifying...")

    try:
        processed_img = preprocess_image(uploaded_file)
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        st.success(f"**Predicted Class:** {class_names.get(predicted_class, 'Unknown')}")
        st.info(f"**Confidence:** {confidence:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
