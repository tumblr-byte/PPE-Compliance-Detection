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

# Page configuration
st.set_page_config(
    page_title="PPE Compliance Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
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
    """Process uploaded image with PPE compliance detection"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model(img_array)
    
    # Get annotated image
    annotated_img = results[0].plot()
    
    return annotated_img, results[0]

def process_video(model, video_path, progress_bar, status_text):
    """Process uploaded video with PPE compliance detection"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run PPE detection
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Write frame to output
        out.write(annotated_frame)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path

def main():
    st.title("PPE Compliance Detection System")
    st.markdown("**Demo Project for Testing** - Detects vest, gloves, and other safety equipment compliance")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load PPE detection model. Please ensure 'best.pt' is in the same directory as this script.")
        return
    
    # Sidebar for mode selection
    st.sidebar.title("Detection Mode")
    mode = st.sidebar.selectbox(
        "Choose input type:",
        ["Image Upload", "Video Upload", "Live Camera"]
    )
    
    if mode == "Image Upload":
        st.header("PPE Image Detection")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image file to detect PPE compliance"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Process image
            with st.spinner("Processing image for PPE compliance..."):
                processed_img, results = process_image(model, image)
            
            with col2:
                st.subheader("PPE Detection Results")
                st.image(processed_img, use_column_width=True)
            
            # Display detection results
            if len(results.boxes) > 0:
                st.subheader("PPE Compliance Summary")
                detected_classes = []
                for i, box in enumerate(results.boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    detected_classes.append(f"{class_name}: {confidence:.2f}")
                
                for detection in detected_classes:
                    st.write(f"- {detection}")
            else:
                st.warning("No PPE equipment detected in the image")
    
    elif mode == "Video Upload":
        st.header("PPE Video Detection")
        
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to detect PPE compliance"
        )
        
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Display original video
            st.subheader("Original Video")
            st.video(uploaded_video)
            
            # Process video button
            if st.button("Process Video for PPE Detection"):
                st.subheader("Processing Video...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video
                output_path = process_video(model, video_path, progress_bar, status_text)
                
                # Display processed video
                st.subheader("PPE Detection Results")
                st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f.read(),
                        file_name="ppe_detected_video.mp4",
                        mime="video/mp4"
                    )
                
                # Clean up temporary files
                os.unlink(video_path)
                os.unlink(output_path)
    
    elif mode == "Live Camera":
        st.header("Live PPE Detection")
        st.markdown("**Note: This is a demo project for testing. Live feed is limited to 5 minutes maximum.**")
        
        # Camera settings
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Camera Controls")
            start_camera = st.button("Start PPE Detection")
            stop_camera = st.button("Stop Detection")
            
            # Display settings
            st.subheader("Display Settings")
            display_width = 640
            display_height = 480
            st.write(f"Resolution: {display_width} x {display_height}")
        
        with col2:
            if start_camera:
                st.subheader("Live PPE Detection Feed")
                
                # Create placeholders
                video_placeholder = st.empty()
                status_placeholder = st.empty()
                time_placeholder = st.empty()
                
                # Initialize camera
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
                
                start_time = time.time()
                max_duration = 300  # 5 minutes in seconds
                
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to access camera")
                            break
                        
                        # Check time limit
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_duration:
                            status_placeholder.warning("Demo time limit reached (5 minutes). This is a demo project for testing.")
                            break
                        
                        # Run PPE detection
                        results = model(frame)
                        annotated_frame = results[0].plot()
                        
                        # Convert BGR to RGB for display
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        video_placeholder.image(
                            annotated_frame_rgb,
                            channels="RGB",
                            width=display_width
                        )
                        
                        # Update time display
                        remaining_time = max_duration - elapsed_time
                        time_placeholder.info(f"Time remaining: {remaining_time:.0f} seconds")
                        
                        # Display current detections
                        if len(results[0].boxes) > 0:
                            current_detections = []
                            for box in results[0].boxes:
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                confidence = float(box.conf[0])
                                current_detections.append(f"{class_name} ({confidence:.2f})")
                            
                            status_placeholder.success(f"PPE Detected: {', '.join(current_detections)}")
                        else:
                            status_placeholder.warning("No PPE equipment detected")
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.1)
                
                except Exception as e:
                    st.error(f"Error during live detection: {str(e)}")
                
                finally:
                    cap.release()
                    video_placeholder.empty()
                    status_placeholder.info("Live detection stopped")
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.subheader("About PPE Detection")
    st.sidebar.info(
        "This system detects Personal Protective Equipment including:\n"
        "- Safety vests\n"
        "- Safety gloves\n"
        "- Hard hats\n"
        "- Other safety equipment"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demo Limitations")
    st.sidebar.warning(
        "This is a demo project for testing purposes:\n"
        "- Live camera limited to 5 minutes\n"
        "- Processing time may vary\n"
        "- Accuracy depends on lighting conditions"
    )

if __name__ == "__main__":
    main()