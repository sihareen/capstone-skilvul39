import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import tensorflow as tf


#pasang login page
def login_page():
    st.title("Login")
    st.write("**Please enter your username and password to access the Smart Lemon Insight app🍋**")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "39" and password == "skilvul":  # Replace with your own credentials
            st.session_state.logged_in = True
            st.experimental_rerun()  # Force a rerun of the app
        else:
            st.error("Invalid username or password")


#st.set_page_config(page_title='lemonQC', page_icon='🍋')
# Page title
def Home():
    st.title('🍋x🤖 Smart Lemon Insight')
    st.divider()
    with st.expander('***About this app***'):
        st.info('*Smart Lemon Insight adalah sebuah sistem kecerdasan buatan yang dirancang untuk menentukan kualitas buah lemon secara otomatis dan presisi. Sistem ini menggunakan teknologi deep learning dengan memanfaatkan Keras dari TensorFldow dan algoritma YOLO (You Only Look Once) untuk melatih model deteksi objek. Dengan KerasCV, Smart Lemon Insight dapat memanfaatkan model YOLOv8 yang telah dilatih sebelumnya untuk mendeteksi dan mengklasifikasikan lemon berdasarkan parameter seperti warna, ukuran, dan ada tidaknya cacat. Proses ini melibatkan pengolahan citra digital yang mencakup augmentasi data, konversi ruang warna, dan ekstraksi fitur. Dengan metode ini, Smart Lemon Insight mampu memberikan penilaian kualitas lemon secara real-time dengan akurasi tinggi, sehingga membantu petani dan produsen lemon dalam memastikan kualitas produk mereka tetap konsisten dan optimal*')
        st.divider()
        st.markdown('**What can this app do?**')
        st.info('Smart Lemon Insight mampu mengklasifikasikan lemon ke dalam beberapa kategori mutu berdasarkan parameter visual seperti warna dan ukuran, memastikan bahwa hanya lemon berkualitas tinggi yang dipasarkan.')
        st.divider()
        st.markdown('**Links:**')
        st.code('''- https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset
        ''', language='markdown')
        st.code('''- hhttps://github.com/sihareen/capstone-skilvul39
        ''', language='markdown')
pass

def Realtime_Detection():
    model1 = YOLO("models/bestv8.pt")  # Use a pre-trained model from Ultralytics
    model1.conf = 0.25  # Set confidence threshold
    model1.iou = 0.45  # Set IoU threshold for NMS

    # Streamlit app
    st.title('🍋x🤖 Smart Lemon Insight')
    st.write("YOLOv8 Object Detection")

    # Input selection
    input_type = st.radio("Select input type", ["Image", "Webcam Realtime", "Video file"])

    if input_type == "Image":
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Detecting objects...")

            # Perform object detection on the uploaded image
            results = model1(image)

            # Process each result in the list
            for result in results:
                # Convert the bounding boxes to a pandas DataFrame
                boxes = result.boxes
                if boxes is not None:
                    df = boxes.xyxy.cpu().numpy()  # Convert to numpy array
                    st.write("Detection Results:", df)  # Display the detection results

                # Plot results
                annotated_image = result.plot()  # Generate image with bounding boxes and labels

                # Display results
                st.image(annotated_image, caption='Detected Image.', use_column_width=True)

    elif input_type == "Webcam Realtime":
        st.warning("NOTE : In order to use this mode, you need to give webcam access. "
                "After clicking 'Start' , it takes about 10-20 seconds to ready the webcam.")

        spinner_message = "Wait a sec, getting some things done..."

        with st.spinner(spinner_message):

            class VideoProcessor:

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")

                    # Detect objects in the frame using your YOLOv8 model
                    results = model1(img)

                    # Annotate the frame with bounding boxes and labels
                    annotated_frame = results[0].plot()

                    frame = av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

                    return frame


            webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                            rtc_configuration=RTCConfiguration(
                                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

    else:
        # Real-time object detection using video file
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            # Process video frames
            cap = cv2.VideoCapture(uploaded_video.name)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects in the frame
                results = model1(frame)

                # Annotate the frame with bounding boxes and labels
                annotated_frame = results[0].plot()

                # Display the annotated frame
                st.image(annotated_frame)
pass

def Lemon_Classification():
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
    st.title('🍋x🤖 Smart Lemon Insight')
    st.write("Lemon Quality Classification with TensorFlow using a pre-trained CNN model")

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
            class_names = ["Lemon with bad quality", "Lemon not detected", "Lemon with good quality"]
            st.write(f'Prediction: {class_names[predicted_class]}')
            st.write(f'Confidence: {confidence:.2f}')
    else:
        detect_with_webcam()
pass

# Create the page functions list
def main_app():
    page_names = ["Home", "Realtime Detection", "Lemon Classification"]
    page_functions = [Home, Realtime_Detection, Lemon_Classification]

    # Create the selectbox
    selected_page = st.selectbox("Select a page 🍋", page_names)

    # Call the selected page function
    page_functions[page_names.index(selected_page)]()

# Create a session state variable to track login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Show the login page if not logged in
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
# Add the developers section
st.divider()
st.markdown('**Developers:**')
st.text('-Muhammad Rizkan Harin Faza')
st.text('-Dicky Wijaya Saputra')
st.text('-Eko Santoso')
st.text('-Nisrina Putri Fernanda Fairuz')
