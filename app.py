import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Reckless Driving Behaviours • YOLO", layout="wide")

st.title("🚗 Reckless Driving Behaviours using YOLOv11")

file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("By **Jhillian Millare Cabos**")

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("yolov11.pt")

model = load_model()

reckless_classes = {1, 2, 4, 5, 6, 7, 9, 10}

def classify_recklessness(class_ids):
    return "Reckless Driving" if any(cid in reckless_classes for cid in class_ids) else "Not Reckless Driving"

if file is not None:
    if file.type.startswith("image"):
        image = np.array(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR))
        results = model(image, conf=confidence, iou=iou_thresh)
        annotated = results[0].plot()
        class_ids = [int(cls) for cls in results[0].boxes.cls]
        label = classify_recklessness(class_ids)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected: {label}", use_container_width=True)
    else:
        tfile = open("temp_video", "wb")
        tfile.write(file.read())
        tfile.close()
        cap = cv2.VideoCapture("temp_video")
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated = results[0].plot()
            class_ids = [int(cls) for cls in results[0].boxes.cls]
            label = classify_recklessness(class_ids)
            font_scale = 1.5 if label == "Reckless Driving" else 0.7
            color = (0, 0, 255) if label == "Reckless Driving" else (0, 255, 0)
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        cap.release()
