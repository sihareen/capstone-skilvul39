import streamlit as st
import cv2
import numpy as np

def load_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        st.error(f"Image not found or failed to load at {image_path}")
        return None
    return frame

def capture_webcam():
    cap = cv2.VideoCapture(1)  # Change the index if necessary
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from webcam")
        return None
    return frame

st.title("Image Processing with OpenCV")

option = st.selectbox("Choose input source", ("Image File", "Webcam"))

if option == "Image File":
    image_path = st.text_input("Enter the path to the image file")
    if st.button("Load Image"):
        frame = load_image(image_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption="Loaded Image", use_column_width=True)

elif option == "Webcam":
    if st.button("Capture Image"):
        frame = capture_webcam()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption="Captured Image", use_column_width=True)