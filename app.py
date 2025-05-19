import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import RTDETR
import tempfile
import os

# Load model
model = RTDETR("best.pt")

# Class names (replace with your actual mapping if different)
model.names = {
    0: 'c0 - Safe Driving',
    1: 'c1 - Texting',
    2: 'c2 - Talking on the phone',
    3: 'c3 - Operating the Radio',
    4: 'c4 - Drinking',
    5: 'c5 - Reaching Behind',
    6: 'c6 - Hair and Makeup',
    7: 'c7 - Talking to Passenger',
    8: 'd0 - Eyes Closed',
    9: 'd1 - Yawning',
    10: 'd2 - Nodding Off',
    11: 'd3 - Eyes Open',
    12: 'e1 - Seat Belt'
}

# Define reckless and safe behavior classes
reckless_classes = {1, 2, 4, 5, 6, 7, 9, 10}
safe_classes = {0}

def classify_recklessness(class_ids):
    if any(cls in reckless_classes for cls in class_ids):
        return "Reckless Driving"
    return "Not Reckless Driving"

# Streamlit GUI
st.set_page_config(page_title="Reckless Driving Behaviours using RT-DETR", layout="wide")
st.title("🚗 Reckless Driving Behaviours using RT-DETR")

file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])
confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold:", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("By Jhillian Millare Cabos")

if file is not None:
    file_type = file.type

    if "image" in file_type:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = model(image_bgr, conf=confidence, iou=iou_thresh)
        annotated_img = results[0].plot()

        class_ids = [int(cls) for cls in results[0].boxes.cls]
        label = classify_recklessness(class_ids)
        st.text(f"Detected: {label}")

        st.image(annotated_img, caption="Detection Result with Classification", use_container_width=True)

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated_frame = results[0].plot()

            class_ids = [int(cls) for cls in results[0].boxes.cls]
            label = classify_recklessness(class_ids)

            # Add label overlay
            font_scale = 1.5 if label == "Reckless Driving" else 0.7
            color = (0, 0, 255) if label == "Reckless Driving" else (0, 255, 0)
            cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
