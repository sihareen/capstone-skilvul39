import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a pre-trained model from Ultralytics
model.conf = 0.25  # Set confidence threshold
model.iou = 0.45  # Set IoU threshold for NMS

# Streamlit app
st.title("YOLOv8 Object Detection")

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

elif input_type == "Webcam Realtime":
    st.warning("NOTE : In order to use this mode, you need to give webcam access. "
               "After clicking 'Start' , it takes about 10-20 seconds to ready the webcam.")

    spinner_message = "Wait a sec, getting some things done..."

    with st.spinner(spinner_message):

        class VideoProcessor:

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Detect objects in the frame using your YOLOv8 model
                results = model(img)

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
            results = model(frame)

            # Annotate the frame with bounding boxes and labels
            annotated_frame = results[0].plot()

            # Display the annotated frame
            st.image(annotated_frame)