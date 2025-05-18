import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Load the model
model = YOLO("yolo24k.pt")  # Your custom model path

# Set page config
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("🚗 Reckless Driving Behaviours using YOLOV11")

# Sidebar for file upload and parameters
file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

# Continuous sliders for confidence and IoU
confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold:", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("YOLOV11")

# Define reckless driving behaviors
reckless_classes = {1, 2, 4, 5, 6, 7, 8 ,9 ,10}  # Texting, Talking on the phone, Drinking, Reaching Behind, Hair and Makeup
safe_classes = {0}  # Safe Driving

def classify_recklessness(class_id):
    """Classify the behavior as 'reckless' or 'not reckless'."""
    if class_id in reckless_classes:
        return "Reckless Driving"
    else:
        return "Not Reckless Driving"

if file is not None:
    file_type = file.type

    if "image" in file_type:
        # Handle image input
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        results = model(image_np, conf=confidence, iou=iou_thresh)
        annotated_img = np.squeeze(results[0].plot())

        # Add recklessness classification to the image
        for result in results[0].boxes:
            class_id = int(result.cls)
            label = classify_recklessness(class_id)
            st.text(f"Detected: {label}")

        st.image(annotated_img, caption="Detection Result with Classification", use_container_width=True)

    elif "video" in file_type:
        # Handle video input
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        video_path = tfile.name

        st.video(video_path)

        cap = cv2.VideoCapture(video_path)
        out_path = os.path.join("outputs", os.path.basename(video_path))
        os.makedirs("outputs", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated_frame = np.squeeze(results[0].plot())

            # Add recklessness classification to each frame
            for result in results[0].boxes:
                class_id = int(result.cls)
                label = classify_recklessness(class_id)
                # Display the classification on the frame
                cv2.putText(annotated_frame, label, (int(result.xyxy[0]), int(result.xyxy[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(annotated_frame)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        out.release()
        st.success("Video processing complete!")

        st.video(out_path)
