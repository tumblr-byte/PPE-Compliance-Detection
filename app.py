import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import torch
from ultralytics import YOLO
from PIL import Image
import io
import threading

# Page configuration
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model
@st.cache_resource
def load_model():
    """Load the PPE detection model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading PPE model: {str(e)}")
        return None

def process_image(model, image):
    """Process image with PPE detection"""
    img_array = np.array(image)
    results = model(img_array, conf=0.5, iou=0.5)
    filtered_results = filter_highest_confidence_per_class(results[0])
    annotated_img = plot_filtered_results(img_array, filtered_results, model.names)
    return annotated_img, filtered_results

def filter_highest_confidence_per_class(results):
    """Filter results to keep only highest confidence detection per class"""
    if len(results.boxes) == 0:
        return []
    
    class_detections = {}
    
    for i, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        if class_id not in class_detections or confidence > class_detections[class_id]['confidence']:
            class_detections[class_id] = {
                'box_idx': i,
                'confidence': confidence,
                'box': box
            }
    
    filtered_boxes = [det['box'] for det in class_detections.values()]
    return filtered_boxes

def plot_filtered_results(image, filtered_boxes, class_names):
    """Plot detection results on image"""
    img_copy = image.copy()
    
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Draw box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw label
        label = f"{class_names[class_id]}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 15), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img_copy

def process_video(model, video_path, progress_bar, status_text):
    """Process uploaded video with PPE detection"""
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.5, iou=0.5)
        filtered_boxes = filter_highest_confidence_per_class(results[0])
        annotated_frame = plot_filtered_results(frame, filtered_boxes, model.names)
        
        out.write(annotated_frame)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path

def main():
    # Clean header
    st.title("PPE Detection System")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load PPE model. Please check 'best.pt' file.")
        return
    
    # Simple mode selection
    mode = st.selectbox(
        "Select Mode:",
        ["Live Detection", "Upload Image", "Upload Video", "Camera Photo"]
    )
    
    if mode == "Live Detection":
        st.subheader("Live PPE Detection")
        
        # Initialize session state
        if 'live_active' not in st.session_state:
            st.session_state.live_active = False
        if 'frames_processed' not in st.session_state:
            st.session_state.frames_processed = 0
        if 'detection_summary' not in st.session_state:
            st.session_state.detection_summary = {}
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = None
        if 'last_detections' not in st.session_state:
            st.session_state.last_detections = []
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🟢 Start Live", type="primary", key="start_live"):
                st.session_state.live_active = True
                st.success("Live detection started!")
                st.rerun()
        
        with col_btn2:
            if st.button("🔴 Stop & Save", key="stop_live"):
                st.session_state.live_active = False
                st.info("Detection stopped - Results saved!")
        
        with col_btn3:
            if st.session_state.live_active:
                st.success("🔴 LIVE - Auto-capturing every 2 seconds")
            else:
                st.info("⏸️ STOPPED")
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Live Camera Feed**")
            
            if st.session_state.live_active:
                # Auto-refreshing camera for continuous detection
                camera_key = f"live_camera_{int(time.time() / 2)}"  # New key every 2 seconds
                
                camera_photo = st.camera_input(
                    "📷 Live Detection Active - Auto-capturing...",
                    key=camera_key,
                    help="Camera will automatically capture and process frames"
                )
                
                if camera_photo is not None:
                    # Process the captured frame
                    image = Image.open(camera_photo)
                    
                    with st.spinner("Processing live frame..."):
                        processed_img, detections = process_image(model, image)
                    
                    # Update session state
                    st.session_state.frames_processed += 1
                    st.session_state.last_frame = processed_img
                    st.session_state.last_detections = detections
                    
                    # Update detection summary
                    if detections:
                        for box in detections:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            
                            if class_name in st.session_state.detection_summary:
                                st.session_state.detection_summary[class_name] += 1
                            else:
                                st.session_state.detection_summary[class_name] = 1
                    
                    # Display processed frame
                    st.image(
                        processed_img, 
                        caption=f"🔴 LIVE Frame #{st.session_state.frames_processed}",
                        use_column_width=True
                    )
                    
                    # Auto-refresh for continuous effect
                    time.sleep(0.5)
                    st.rerun()
                
                else:
                    st.info("📷 Waiting for camera - Click the camera button to start live detection")
            
            else:
                # Show last captured frame when stopped
                if st.session_state.last_frame is not None:
                    st.image(
                        st.session_state.last_frame,
                        caption="Last Detection Frame (Saved)",
                        use_column_width=True
                    )
                else:
                    # Placeholder
                    placeholder = np.full((400, 600, 3), 220, dtype=np.uint8)
                    cv2.putText(placeholder, "Click 'Start Live' to begin", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    cv2.putText(placeholder, "continuous PPE detection", (160, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    st.image(placeholder, use_column_width=True)
        
        with col2:
            st.write("**Detection Results**")
            
            # Current stats
            st.metric("Frames Processed", st.session_state.frames_processed)
            
            # Current detections
            if st.session_state.live_active and st.session_state.last_detections:
                st.write("**Current Frame:**")
                for box in st.session_state.last_detections:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"• {class_name}: {confidence:.2f}")
            
            # Detection summary
            if st.session_state.detection_summary:
                st.write("---")
                st.write("**Total Detected:**")
                total_items = sum(st.session_state.detection_summary.values())
                st.metric("Total PPE Items", total_items)
                
                for item, count in st.session_state.detection_summary.items():
                    st.write(f"• {item}: {count} times")
            else:
                st.info("No detections yet")
            
            # Download and reset
            if st.session_state.last_frame is not None:
                st.write("---")
                
                # Download button
                img_bytes = io.BytesIO()
                Image.fromarray(st.session_state.last_frame).save(img_bytes, format='PNG')
                
                st.download_button(
                    label="📥 Download Last Frame",
                    data=img_bytes.getvalue(),
                    file_name=f"ppe_detection_{int(time.time())}.png",
                    mime="image/png"
                )
            
            # Reset button
            if st.button("🔄 Reset All"):
                st.session_state.frames_processed = 0
                st.session_state.detection_summary = {}
                st.session_state.last_frame = None
                st.session_state.last_detections = []
                st.session_state.live_active = False
                st.success("Statistics reset!")
                st.rerun()
        
        # Instructions
        if st.session_state.live_active:
            st.info("📹 **Live Mode Active:** Camera will auto-capture every 2 seconds for continuous PPE detection. Click 'Stop & Save' when finished.")
        else:
            st.info("▶️ **How to use:** Click 'Start Live' → Allow camera access → Camera will automatically capture and process frames continuously → Click 'Stop & Save' when done")
    
    elif mode == "Upload Image":
        st.subheader("Image PPE Detection")
        
        uploaded_file = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original**")
                st.image(image, use_column_width=True)
            
            with col2:
                with st.spinner("Processing..."):
                    processed_img, detections = process_image(model, image)
                
                st.write("**Detection Results**")
                st.image(processed_img, use_column_width=True)
                
                if detections:
                    st.success(f"Found {len(detections)} PPE items:")
                    for box in detections:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        st.write(f"• {class_name}: {confidence:.2f}")
                else:
                    st.warning("No PPE detected")
    
    elif mode == "Upload Video":
        st.subheader("Video PPE Detection")
        
        uploaded_video = st.file_uploader("Choose video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Video**")
                st.video(uploaded_video)
            
            with col2:
                if st.button("Process Video"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    output_path = process_video(model, video_path, progress_bar, status_text)
                    
                    st.write("**Processed Video**")
                    st.video(output_path)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download Processed Video",
                            data=f.read(),
                            file_name="ppe_processed_video.mp4",
                            mime="video/mp4"
                        )
                    
                    os.unlink(video_path)
                    os.unlink(output_path)
    
    elif mode == "Camera Photo":
        st.subheader("Camera PPE Detection")
        
        camera_photo = st.camera_input("Take a photo")
        
        if camera_photo:
            image = Image.open(camera_photo)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Captured Photo**")
                st.image(image, use_column_width=True)
            
            with col2:
                with st.spinner("Processing..."):
                    processed_img, detections = process_image(model, image)
                
                st.write("**Detection Results**")
                st.image(processed_img, use_column_width=True)
                
                if detections:
                    st.success(f"Detected {len(detections)} PPE items:")
                    for box in detections:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        st.write(f"• {class_name}: {confidence:.2f}")
                else:
                    st.warning("No PPE equipment detected")

if __name__ == "__main__":
    main()
