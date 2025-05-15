import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your custom model path if needed

st.set_page_config(page_title="Object Detection App using YOLO11", layout="wide")
st.title("🚗 Object Detection App (YOLO11)")

# Sidebar controls
st.sidebar.header("Upload File")
file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

# Custom select slider with specific thresholds (percent)
confidence_percent = st.sidebar.select_slider(
    "Confidence Threshold (%)",
    options=[0, 20, 50, 70, 95],
    value=50
)
confidence = confidence_percent / 100

iou_percent = st.sidebar.select_slider(
    "IoU Threshold (%)",
    options=[0, 20, 50, 70, 95],
    value=50
)
iou_thresh = iou_percent / 100

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit and YOLO11")

# Main logic
if file is not None:
    file_type = file.type

    if "image" in file_type:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to NumPy for YOLOv8
        image_np = np.array(image)
        results = model(image_np, conf=confidence, iou=iou_thresh)
        annotated_img = np.squeeze(results[0].plot())

        st.image(annotated_img, caption="Detection Result", use_column_width=True)

    elif "video" in file_type:
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
            out.write(annotated_frame)

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        out.release()
        st.success("Video processing complete!")

        st.video(out_path)
