import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("models/bestv8.pt")  # Use a pre-trained model from Ultralytics
model.conf = 0.25  # Set confidence threshold
model.iou = 0.45  # Set IoU threshold for NMS

# Streamlit app
st.title("YOLOv8 Object Detection")
st.write("Upload an image to detect objects")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    # Perform object detection
    results = model(image)

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