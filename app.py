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
import queue

# Page configuration
st.set_page_config(
    page_title="PPE Compliance Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for webcam
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'webcam_thread' not in st.session_state:
    st.session_state.webcam_thread = None
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=2)

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
    
    # Run inference with confidence threshold and NMS
    results = model(img_array, conf=0.5, iou=0.5)
    
    # Filter to keep only highest confidence per class
    filtered_results = filter_highest_confidence_per_class(results[0])
    
    # Get annotated image with filtered results
    annotated_img = plot_filtered_results(img_array, filtered_results, model.names, is_video=False)
    
    return annotated_img, filtered_results

def filter_highest_confidence_per_class(results):
    """Filter results to keep only highest confidence detection per class"""
    if len(results.boxes) == 0:
        return results
    
    # Group detections by class
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
    
    # Create filtered results
    filtered_boxes = [det['box'] for det in class_detections.values()]
    return filtered_boxes

def plot_filtered_results(image, filtered_boxes, class_names, is_video=False):
    """Plot filtered results on image with enhanced visibility for video"""
    img_copy = image.copy()
    
    # Set different parameters for video vs image
    if is_video:
        # Enhanced visibility for video
        box_thickness = 4
        font_scale = 1.2
        font_thickness = 3
        label_padding = 20
    else:
        # Normal settings for image
        box_thickness = 2
        font_scale = 0.6
        font_thickness = 2
        label_padding = 10
    
    for box in filtered_boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Draw bounding box with enhanced thickness for video
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        
        # Draw label with larger font for video
        label = f"{class_names[class_id]}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Background rectangle for label
        cv2.rectangle(
            img_copy, 
            (x1, y1 - label_size[1] - label_padding), 
            (x1 + label_size[0] + 15, y1), 
            (0, 255, 0), 
            -1
        )
        
        # Text label
        cv2.putText(
            img_copy, 
            label, 
            (x1 + 5, y1 - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0, 0, 0), 
            font_thickness
        )
    
    return img_copy

def webcam_capture_thread(model, frame_queue, stop_event, display_size=(500, 500)):
    """Background thread for webcam capture and processing"""
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
    
    # Set buffer size to 1 to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    start_time = time.time()
    max_duration = 300  # 5 minutes
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > max_duration:
                break
            
            # Resize frame to exact display size
            frame_resized = cv2.resize(frame, display_size)
            
            # Run PPE detection
            try:
                results = model(frame_resized, conf=0.5, iou=0.5)
                filtered_boxes = filter_highest_confidence_per_class(results[0])
                annotated_frame = plot_filtered_results(frame_resized, filtered_boxes, model.names, is_video=True)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Put frame in queue (non-blocking)
                detection_info = {
                    'frame': frame_rgb,
                    'detections': filtered_boxes,
                    'elapsed_time': elapsed_time,
                    'remaining_time': max_duration - elapsed_time
                }
                
                if not frame_queue.full():
                    try:
                        frame_queue.put_nowait(detection_info)
                    except queue.Full:
                        pass
                        
            except Exception as e:
                print(f"Detection error: {e}")
                continue
            
            time.sleep(0.05)  # Reduce CPU usage
            
    except Exception as e:
        print(f"Webcam capture error: {e}")
    finally:
        cap.release()

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
        
        # Run PPE detection with confidence threshold
        results = model(frame, conf=0.5, iou=0.5)
        
        # Filter to highest confidence per class
        filtered_boxes = filter_highest_confidence_per_class(results[0])
        
        # Plot filtered results with enhanced visibility for video
        annotated_frame = plot_filtered_results(frame, filtered_boxes, model.names, is_video=True)
        
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
                processed_img, filtered_results = process_image(model, image)
            
            with col2:
                st.subheader("PPE Detection Results")
                st.image(processed_img, use_column_width=True)
            
            # Display detection results
            if len(filtered_results) > 0:
                st.subheader("PPE Compliance Summary")
                for box in filtered_results:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"- {class_name}: {confidence:.2f}")
            else:
                st.warning("No PPE equipment detected in the image (confidence threshold: 0.5)")
    
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
            st.video(uploaded_video, width=400)
            
            # Process video button
            if st.button("Process Video for PPE Detection"):
                st.subheader("Processing Video...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video
                output_path = process_video(model, video_path, progress_bar, status_text)
                
                # Display processed video
                st.subheader("PPE Detection Results")
                st.video(output_path, width=400)
                
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
        st.header("Live PPE Detection (500x500px)")
        st.markdown("**Note: This is a demo project for testing. Live feed is limited to 5 minutes maximum.**")
        
        # Create layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Camera Controls")
            
            # Start camera button
            if st.button("🎥 Start PPE Detection", key="start_btn"):
                if not st.session_state.webcam_running:
                    st.session_state.webcam_running = True
                    # Create stop event for thread
                    st.session_state.stop_event = threading.Event()
                    # Start webcam thread
                    st.session_state.webcam_thread = threading.Thread(
                        target=webcam_capture_thread,
                        args=(model, st.session_state.frame_queue, st.session_state.stop_event, (500, 500))
                    )
                    st.session_state.webcam_thread.daemon = True
                    st.session_state.webcam_thread.start()
                    st.rerun()
            
            # Stop camera button
            if st.button("🛑 Stop Detection", key="stop_btn"):
                if st.session_state.webcam_running:
                    st.session_state.webcam_running = False
                    if hasattr(st.session_state, 'stop_event'):
                        st.session_state.stop_event.set()
                    # Clear the queue
                    while not st.session_state.frame_queue.empty():
                        try:
                            st.session_state.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    st.rerun()
            
            # Display settings
            st.subheader("Display Settings")
            st.write("📐 Resolution: 500 x 500 pixels")
            st.write("⏱️ Max Duration: 5 minutes")
            st.write(f"📊 Status: {'🟢 Running' if st.session_state.webcam_running else '🔴 Stopped'}")
        
        with col2:
            st.subheader("Live PPE Detection Feed")
            
            # Create placeholders
            video_placeholder = st.empty()
            detection_placeholder = st.empty()
            time_placeholder = st.empty()
            
            if st.session_state.webcam_running:
                # Display live feed
                try:
                    # Get latest frame from queue
                    detection_info = st.session_state.frame_queue.get_nowait()
                    
                    # Display the frame with fixed size
                    video_placeholder.image(
                        detection_info['frame'],
                        caption="Live PPE Detection - 500x500px",
                        width=500,
                        channels="RGB"
                    )
                    
                    # Display detection results
                    if len(detection_info['detections']) > 0:
                        detections_text = []
                        for box in detection_info['detections']:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            detections_text.append(f"✅ {class_name} ({confidence:.2f})")
                        
                        detection_placeholder.success("**PPE Detected:**\n\n" + "\n".join(detections_text))
                    else:
                        detection_placeholder.warning("⚠️ No PPE equipment detected")
                    
                    # Display time remaining
                    remaining_time = max(0, detection_info['remaining_time'])
                    if remaining_time > 0:
                        minutes = int(remaining_time // 60)
                        seconds = int(remaining_time % 60)
                        time_placeholder.info(f"⏰ Time remaining: {minutes}:{seconds:02d}")
                    else:
                        time_placeholder.error("⏰ Demo time limit reached!")
                        st.session_state.webcam_running = False
                        if hasattr(st.session_state, 'stop_event'):
                            st.session_state.stop_event.set()
                    
                    # Auto-refresh every 100ms for smooth video
                    time.sleep(0.1)
                    st.rerun()
                    
                except queue.Empty:
                    # No new frame available, show loading
                    video_placeholder.info("📹 Initializing camera... Please wait.")
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error displaying webcam feed: {str(e)}")
                    st.session_state.webcam_running = False
                    
            else:
                # Show placeholder when not running
                placeholder_img = np.full((500, 500, 3), 128, dtype=np.uint8)
                video_placeholder.image(
                    placeholder_img,
                    caption="Click 'Start PPE Detection' to begin live detection",
                    width=500,
                    channels="RGB"
                )
                detection_placeholder.info("🎯 Ready for PPE detection")
                time_placeholder.info("⏰ Timer: Ready")
    
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
    st.sidebar.subheader("Live Camera Features")
    st.sidebar.success(
        "✅ **Improved Live Detection:**\n"
        "- 500x500px display resolution\n"
        "- Enhanced visibility with larger text\n"
        "- Real-time PPE compliance alerts\n"
        "- Smooth video streaming\n"
        "- Background processing for better performance"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demo Limitations")
    st.sidebar.warning(
        "This is a demo project for testing purposes:\n"
        "- Live camera limited to 5 minutes\n"
        "- Processing time may vary\n"
        "- Accuracy depends on lighting conditions\n"
        "- Requires webcam permissions"
    )

if __name__ == "__main__":
    main()
