import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Memuat model yang telah dilatih
@st.cache_data
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

# Fungsi untuk menjalankan deteksi secara real-time menggunakan webcam
def detect_with_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Could not open webcam.")
            break

        # Melakukan prediksi pada frame saat ini
        predicted_class, confidence = predict(frame, model)
        
        # Menampilkan prediksi pada frame
        class_names = ["Lemon with bad quality", "Lemon not detected", "Lemon with good quality"]
        label = f'{class_names[predicted_class]} ({confidence:.2f})'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Menampilkan frame pada Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

# Streamlit app
st.title("Lemon Quality Classification with TensorFlow")
st.write("Using a pre-trained CNN model")

# Pilihan untuk mengunggah gambar atau menggunakan webcam
option = st.selectbox("Choose input method", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
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
        class_names = ["Lemon with good quality", "Lemon not detected", "Lemon with bad quality"]
        st.write(f'Prediction: {class_names[predicted_class]}')
        st.write(f'Confidence: {confidence:.2f}')
else:
    detect_with_webcam()
