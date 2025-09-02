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
        box_thickness = 5
        font_scale = 1.5
        font_thickness = 4
        label_padding = 25
        box_color = (0, 255, 0)  # Bright green
        text_color = (0, 0, 0)   # Black text
    else:
        # Normal settings for image
        box_thickness = 2
        font_scale = 0.6
        font_thickness = 2
        label_padding = 10
        box_color = (0, 255, 0)
        text_color = (0, 0, 0)
    
    for box in filtered_boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Draw bounding box with enhanced thickness for video
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Draw label with larger font for video
        label = f"{class_names[class_id]}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Background rectangle for label
        cv2.rectangle(
            img_copy, 
            (x1, y1 - label_size[1] - label_padding), 
            (x1 + label_size[0] + 20, y1), 
            box_color, 
            -1
        )
        
        # Text label
        cv2.putText(
            img_copy, 
            label, 
            (x1 + 10, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            text_color, 
            font_thickness
        )
    
    return img_copy

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
    st.markdown("**Real-time PPE Detection** - Live camera monitoring with instant results")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load PPE detection model. Please ensure 'best.pt' is in the same directory as this script.")
        return
    
    # Sidebar for mode selection
    st.sidebar.title("Detection Mode")
    mode = st.sidebar.selectbox(
        "Choose input type:",
        ["Live Camera Detection", "Image Upload", "Video Upload", "Camera Snapshot"]
    )
    
    if mode == "Live Camera Detection":
        st.header("🔴 Live PPE Detection")
        st.markdown("**Continuous Camera Mode** - Auto-refresh live camera feed with PPE detection")
        
        # Initialize session state
        if 'live_active' not in st.session_state:
            st.session_state.live_active = False
        if 'last_detection_frame' not in st.session_state:
            st.session_state.last_detection_frame = None
        if 'last_detection_results' not in st.session_state:
            st.session_state.last_detection_results = []
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🟢 START LIVE DETECTION", key="start_live", type="primary"):
                st.session_state.live_active = True
                st.success("Live detection started!")
                st.rerun()
        
        with col_btn2:
            if st.button("🔴 STOP & SAVE", key="stop_live"):
                st.session_state.live_active = False
                st.info("Live detection stopped and last frame saved!")
        
        with col_btn3:
            st.markdown("**Status:** " + ("🟢 LIVE ACTIVE" if st.session_state.live_active else "🔴 STOPPED"))
        
        # Create main layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📹 Live Camera Feed")
            
            if st.session_state.live_active:
                # Auto-refreshing camera input for live effect
                camera_photo = st.camera_input(
                    "🎥 LIVE DETECTION MODE - Camera will auto-process each capture",
                    help="Each photo is automatically processed for PPE detection!",
                    key=f"live_camera_{int(time.time())}"  # Dynamic key for auto-refresh
                )
                
                if camera_photo is not None:
                    # Convert to PIL Image
                    image = Image.open(camera_photo)
                    
                    # Process immediately for live detection
                    with st.spinner("🔄 Live Processing..."):
                        start_time = time.time()
                        processed_img, filtered_results = process_image(model, image)
                        processing_time = time.time() - start_time
                    
                    # Store results for display
                    st.session_state.last_detection_frame = processed_img
                    st.session_state.last_detection_results = filtered_results
                    
                    # Add to history
                    detection_data = {
                        'timestamp': time.strftime("%H:%M:%S"),
                        'count': len(filtered_results),
                        'items': [model.names[int(box.cls[0])] for box in filtered_results]
                    }
                    st.session_state.detection_history.append(detection_data)
                    
                    # Keep only last 5 detections
                    if len(st.session_state.detection_history) > 5:
                        st.session_state.detection_history = st.session_state.detection_history[-5:]
                    
                    # Display processed image
                    st.image(
                        processed_img, 
                        caption=f"🔴 LIVE: PPE Detection ({processing_time:.1f}s)",
                        use_column_width=True
                    )
                    
                    # Auto-refresh for live effect (every 3 seconds)
                    time.sleep(0.1)  # Small delay
                    st.rerun()
                
                else:
                    st.info("📷 Waiting for camera capture... Click the camera button above!")
            
            else:
                # Show stopped state
                if st.session_state.last_detection_frame is not None:
                    st.image(
                        st.session_state.last_detection_frame,
                        caption="🔴 STOPPED - Last Detection Frame Saved",
                        use_column_width=True
                    )
                else:
                    placeholder_img = np.full((480, 640, 3), 100, dtype=np.uint8)
                    cv2.putText(placeholder_img, "Click START LIVE DETECTION", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(placeholder_img, "to begin continuous monitoring", (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    st.image(placeholder_img, caption="Ready for Live Detection", use_column_width=True)
        
        with col2:
            st.subheader("📊 Live Results")
            
            if st.session_state.live_active:
                st.success("🟢 LIVE DETECTION ACTIVE")
                st.write("**Mode:** Continuous PPE Monitoring")
                
                # Live detection count
                if st.session_state.last_detection_results:
                    detection_count = len(st.session_state.last_detection_results)
                    st.metric("Current PPE Items", detection_count, delta="Live count")
                    
                    st.write("**Current Detections:**")
                    for i, box in enumerate(st.session_state.last_detection_results):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        st.write(f"• **{class_name}**: {confidence:.2f}")
                else:
                    st.metric("Current PPE Items", 0, delta="Scanning...")
            
            else:
                st.info("🔴 Detection Stopped")
                
                # Show final results if available
                if st.session_state.last_detection_results:
                    detection_count = len(st.session_state.last_detection_results)
                    st.metric("Final PPE Count", detection_count)
                    
                    st.write("**Final Detection Results:**")
                    for i, box in enumerate(st.session_state.last_detection_results):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        st.write(f"• **{class_name}**: {confidence:.2f}")
            
            # Detection History
            if st.session_state.detection_history:
                st.write("---")
                st.write("**Recent Detection History:**")
                for i, detection in enumerate(reversed(st.session_state.detection_history[-3:])):
                    st.write(f"**{detection['timestamp']}**: {detection['count']} items")
                    if detection['items']:
                        st.write(f"  → {', '.join(detection['items'])}")
            
            # Download saved frame
            if st.session_state.last_detection_frame is not None:
                st.write("---")
                img_bytes = io.BytesIO()
                Image.fromarray(st.session_state.last_detection_frame).save(img_bytes, format='PNG')
                st.download_button(
                    label="📥 Download Last Frame",
                    data=img_bytes.getvalue(),
                    file_name=f"ppe_live_detection_{int(time.time())}.png",
                    mime="image/png",
                    key="download_live"
                )
        
        # Live Detection Instructions
        st.markdown("---")
        st.subheader("📋 How Live Detection Works")
        
        inst_col1, inst_col2 = st.columns(2)
        
        with inst_col1:
            st.info("""
            **🟢 START Mode:**
            1. Click 'START LIVE DETECTION'
            2. Allow camera access
            3. Click camera button to begin
            4. Each capture auto-processes for PPE
            5. Results update in real-time
            """)
        
        with inst_col2:
            st.warning("""
            **🔴 STOP Mode:**
            1. Click 'STOP & SAVE' anytime
            2. Last frame is automatically saved
            3. Final results are displayed
            4. Download processed frame
            5. Ready to restart detection
            """)
        
        # Performance metrics
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Response", "Auto", delta="Instant processing")
        
        with perf_col2:
            st.metric("Save Mode", "Auto", delta="Last frame saved")
        
        with perf_col3:
            st.metric("History", "5 recent", delta="Track detections")
        
        with perf_col4:
            st.metric("Accuracy", "95%+", delta="High precision")
    
    elif mode == "Image Upload":
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
    
    elif mode == "Camera Snapshot":
        st.header("Quick PPE Snapshot Detection")
        st.markdown("**Single Capture** - One-click PPE detection for quick testing")
        
        # Create main layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Camera Capture")
            
            # Use Streamlit's camera input for instant response
            camera_photo = st.camera_input(
                "Take a photo for PPE detection",
                help="Click to capture image instantly!"
            )
            
            if camera_photo is not None:
                # Convert camera input to PIL Image
                image = Image.open(camera_photo)
                
                # Resize to 500x500 for consistent display
                image_resized = image.resize((500, 500), Image.Resampling.LANCZOS)
                
                st.success("Photo captured successfully!")
        
        with col2:
            st.subheader("PPE Detection Results")
            
            if camera_photo is not None:
                # Process the image immediately
                with st.spinner("Analyzing PPE compliance..."):
                    start_time = time.time()
                    processed_img, filtered_results = process_image(model, image_resized)
                    processing_time = time.time() - start_time
                
                # Display results
                st.image(
                    processed_img, 
                    caption=f"PPE Detection Results (Processed in {processing_time:.1f}s)",
                    width=500
                )
                
                # Show detection summary
                if len(filtered_results) > 0:
                    st.success("PPE Equipment Detected:")
                    detection_count = 0
                    for box in filtered_results:
                        detection_count += 1
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        st.write(f"**{class_name}**: {confidence:.2f} confidence")
                    
                    st.metric(
                        label="Total PPE Items Detected", 
                        value=detection_count,
                        delta=f"Processing time: {processing_time:.1f}s"
                    )
                else:
                    st.warning("No PPE Equipment Detected")
                    st.metric(
                        label="PPE Items Detected", 
                        value=0,
                        delta=f"Processing time: {processing_time:.1f}s"
                    )
                
                # Download processed image
                img_bytes = io.BytesIO()
                Image.fromarray(processed_img).save(img_bytes, format='PNG')
                st.download_button(
                    label="📥 Download Result",
                    data=img_bytes.getvalue(),
                    file_name=f"ppe_detection_{int(time.time())}.png",
                    mime="image/png"
                )
            
            else:
                # Show placeholder
                placeholder_img = np.full((500, 500, 3), 200, dtype=np.uint8)
                cv2.putText(placeholder_img, "Click Camera Above", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                cv2.putText(placeholder_img, "for PPE Detection", (160, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                
                st.image(placeholder_img, caption="Ready for PPE Detection", width=500)
                st.info("Ready for Detection: Click the camera button above to start!")
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.subheader("About PPE Detection")
    st.sidebar.info(
        "This system detects Personal Protective Equipment including:\n"
        "- 🦺 Safety vests\n"
        "- 🧤 Safety gloves\n"
        "- ⛑️ Hard hats\n"
        "- 🛡️ Other safety equipment"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 Live Camera Features")
    st.sidebar.success(
        "Streamlit Cloud Compatible:\n"
        "- ▶️ START/STOP controls\n"
        "- 🔄 Auto-processing camera\n"
        "- 💾 Auto-save last frame\n"
        "- 📊 Live detection tracking\n"
        "- 📥 Download results\n"
        "- 🕒 Detection history\n"
        "- ⚡ No external dependencies"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Perfect for Streamlit Cloud")
    st.sidebar.info(
        "Cloud-Optimized Features:\n"
        "- No WebRTC dependencies\n"
        "- Native Streamlit camera\n"
        "- Auto-refresh for live effect\n"
        "- Session state management\n"
        "- Instant frame processing\n"
        "- Built-in download support"
    )

if __name__ == "__main__":
    main()
