import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(page_title="PPE Detection Demo", layout="wide")

st.title("PPE Detection System")
st.markdown("Upload an image or video below to see how the system detects PPE items such as helmets and safety vests. The detection will display bounding boxes and class labels.")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()

# Utility functions
def filter_highest_confidence_per_class(results):
    if len(results.boxes) == 0:
        return []
    class_detections = {}
    for i, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id not in class_detections or confidence > class_detections[class_id]['confidence']:
            class_detections[class_id] = {'box_idx': i, 'confidence': confidence, 'box': box}
    return [det['box'] for det in class_detections.values()]

def plot_filtered_results(image, filtered_boxes, class_names):
    img_copy = image.copy()
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 8, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    return img_copy

def process_image(image):
    img_array = np.array(image)
    results = model(img_array, conf=0.5, iou=0.5)
    filtered = filter_highest_confidence_per_class(results[0])
    annotated = plot_filtered_results(img_array, filtered, model.names)
    return annotated, filtered

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.5, iou=0.5)
        filtered = filter_highest_confidence_per_class(results[0])
        annotated = plot_filtered_results(frame, filtered, model.names)
        out.write(annotated)

    cap.release()
    out.release()
    return output_path

# Mode selection
mode = st.selectbox("Select Mode", ["Upload Image", "Upload Video"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose image", type=['png','jpg','jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        with st.spinner("Processing image..."):
            annotated_img, detections = process_image(image)
        st.subheader("Detection Results")
        st.image(annotated_img, use_column_width=True)
        if detections:
            st.write(f"Detected {len(detections)} PPE items:")
            for box in detections:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                st.write(f"• {class_name}: {confidence:.2f}")
        else:
            st.write("No PPE detected.")

elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Choose video", type=['mp4','avi','mov'])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        st.subheader("Original Video")
        st.video(uploaded_video)
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                output_path = process_video(video_path)
            st.subheader("Processed Video")
            st.video(output_path)
            with open(output_path, 'rb') as f:
                st.download_button("Download Processed Video", data=f.read(), file_name="ppe_processed_video.mp4", mime="video/mp4")
            os.unlink(video_path)



