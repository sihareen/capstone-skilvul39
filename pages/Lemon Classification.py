import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from roboflow import Roboflow
import tensorflow as tf
import cv2

# Memuat model yang telah dilatih
st.cache_resource.clear()
def load_model():
    model_path = "models/best_model_at_epoch_13.keras"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Fungsi untuk melakukan prediksi
def predict(image, model):
    # Preprocessing image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (300, 300))  # Sesuaikan ukuran dengan input model
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_resized, 0), dtype=tf.float32)
    
    # Melakukan prediksi
    predictions = model.predict(input_tensor)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    
    return predicted_class[0], confidence[0]

# Streamlit app
st.title("Lemon Quality Classification with TensorFlow")
st.write("Using a trained CNN model")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Make predictions
    predicted_class, confidence = predict(image, model)
    
    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")
    
    # Display the prediction
    class_names = ["Lemon with bad quality", "Lemon not detected", "Lemon with good quality"]
    st.write(f'Prediction: {class_names[predicted_class]}')
    st.write(f'Confidence: {confidence:.2f}')
    st.write("Mahluk paling ganteng seluruh semesta : Rizkan Harin")

