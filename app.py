import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

model = YOLO("yolo24k.pt")

st.set_page_config(page_title="Reckless Driving Behaviours using YOLOV11", layout="wide")
st.title("🚗 Reckless Driving Behaviours using YOLOV11")

file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])
confidence = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold:", 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("---")
st.sidebar.markdown("By Jhillian Millare Cabos")

reckless_classes = {1, 2, 4, 5, 6, 7, 9, 10}
safe_classes = {0}

def classify_recklessness(class_ids):
    return "Reckless Driving" if any(cls in reckless_classes for cls in class_ids) else "Not Reckless Driving"

if file is not None:
    file_type = file.type

    if "image" in file_type:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        results = model(np.array(image), conf=confidence, iou=iou_thresh)
        class_ids = [int(result.cls) for result in results[0].boxes]
        label = classify_recklessness(class_ids)
        st.text(f"Detected: {label}")
        annotated_img = np.squeeze(results[0].plot())
        st.image(annotated_img, caption="Detection Result with Classification", use_container_width=True)

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        all_class_ids = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))
            results = model(frame, conf=confidence, iou=iou_thresh)
            class_ids = [int(result.cls) for result in results[0].boxes]
            all_class_ids.extend(class_ids)

            for result in results[0].boxes:
                cls = int(result.cls)
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label = model.names[cls]
                color = (0, 0, 255) if cls in reckless_classes else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        label = classify_recklessness(all_class_ids)
        st.success(f"Final Video Status: {label}")
