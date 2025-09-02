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
import base64

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

def get_live_webcam_component():
    """Return HTML component for live webcam"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { 
                margin: 0; 
                padding: 15px; 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                background: #ffffff;
            }
            .container { 
                max-width: 700px; 
                margin: 0 auto; 
                text-align: center; 
            }
            .controls { 
                margin: 15px 0; 
            }
            button { 
                padding: 12px 24px; 
                margin: 8px; 
                font-size: 14px; 
                font-weight: 500; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer; 
                transition: all 0.2s; 
            }
            #startBtn { 
                background: #28a745; 
                color: white; 
            }
            #stopBtn { 
                background: #dc3545; 
                color: white; 
            }
            button:hover { 
                opacity: 0.9; 
                transform: translateY(-1px); 
            }
            #video { 
                width: 100%; 
                max-width: 640px; 
                border: 2px solid #dee2e6; 
                border-radius: 8px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
            }
            #canvas { 
                display: none; 
            }
            #status { 
                font-size: 14px; 
                font-weight: 500; 
                margin: 15px 0; 
                padding: 10px 16px; 
                border-radius: 6px; 
                background: #f8f9fa; 
                border: 1px solid #dee2e6;
            }
            .live-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                background: #dc3545;
                border-radius: 50%;
                margin-right: 6px;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div id="status">Ready to start detection</div>
            
            <div class="controls">
                <button id="startBtn">Start Live Detection</button>
                <button id="stopBtn">Stop & Save</button>
            </div>
            
            <video id="video" autoplay muted playsinline></video>
            <canvas id="canvas"></canvas>
        </div>
        
        <script>
        let video, canvas, context;
        let isStreaming = false;
        let streamInterval;
        let frameCount = 0;

        document.getElementById('startBtn').addEventListener('click', startWebcam);
        document.getElementById('stopBtn').addEventListener('click', stopWebcam);

        async function startWebcam() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: 'environment' } 
                });
                
                video = document.getElementById('video');
                canvas = document.getElementById('canvas');
                context = canvas.getContext('2d');
                
                video.srcObject = stream;
                video.play();
                
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    isStreaming = true;
                    document.getElementById('status').innerHTML = 
                        '<span class="live-indicator"></span>Live detection active';
                    captureFrames();
                });
                
            } catch (err) {
                document.getElementById('status').innerText = 'Camera access denied';
            }
        }

        function stopWebcam() {
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            if (streamInterval) clearInterval(streamInterval);
            
            isStreaming = false;
            document.getElementById('status').innerText = 'Detection stopped - Frame saved';
        }

        function captureFrames() {
            if (!isStreaming) return;
            
            streamInterval = setInterval(() => {
                if (isStreaming && video.readyState === video.HAVE_ENOUGH_DATA) {
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    canvas.toBlob(blob => {
                        if (blob) {
                            const reader = new FileReader();
                            reader.onload = () => {
                                const base64Data = reader.result.split(',')[1];
                                frameCount++;
                                
                                // Send to Streamlit parent
                                window.parent.postMessage({
                                    type: 'streamlit:setComponentValue',
                                    value: {
                                        frame: base64Data,
                                        count: frameCount
                                    }
                                }, '*');
                            };
                            reader.readAsDataURL(blob);
                        }
                    }, 'image/jpeg', 0.8);
                }
            }, 500);
        }
        </script>
    </body>
    </html>
    """

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
        if 'frames_processed' not in st.session_state:
            st.session_state.frames_processed = 0
        if 'total_detections' not in st.session_state:
            st.session_state.total_detections = 0
        if 'detection_summary' not in st.session_state:
            st.session_state.detection_summary = {}
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = None
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Live webcam component
            import streamlit.components.v1 as components
            
            frame_data = components.html(
                get_live_webcam_component(),
                height=600,
                scrolling=False
            )
            
            # Process incoming frames
            if frame_data and isinstance(frame_data, dict):
                try:
                    if 'frame' in frame_data:
                        # Decode frame
                        image_bytes = base64.b64decode(frame_data['frame'])
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Process with PPE detection
                        processed_img, detections = process_image(model, image)
                        
                        # Update statistics
                        st.session_state.frames_processed = frame_data.get('count', 0)
                        st.session_state.last_frame = processed_img
                        
                        if detections:
                            st.session_state.total_detections += len(detections)
                            
                            # Update detection summary
                            for box in detections:
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                
                                if class_name in st.session_state.detection_summary:
                                    st.session_state.detection_summary[class_name] += 1
                                else:
                                    st.session_state.detection_summary[class_name] = 1
                        
                        # Display processed frame
                        st.image(processed_img, caption=f"Frame {st.session_state.frames_processed}")
                        
                except Exception as e:
                    st.error(f"Processing error: {e}")
        
        with col2:
            st.subheader("Detection Summary")
            
            # Statistics
            st.metric("Frames Processed", st.session_state.frames_processed)
            st.metric("Total Detections", st.session_state.total_detections)
            
            # Detection breakdown
            if st.session_state.detection_summary:
                st.write("**Detected Items:**")
                for item, count in st.session_state.detection_summary.items():
                    st.write(f"• {item}: {count}")
            else:
                st.info("No detections yet")
            
            # Download last frame
            if st.session_state.last_frame is not None:
                st.write("---")
                img_bytes = io.BytesIO()
                Image.fromarray(st.session_state.last_frame).save(img_bytes, format='PNG')
                
                st.download_button(
                    label="Download Last Frame",
                    data=img_bytes.getvalue(),
                    file_name=f"ppe_detection_{int(time.time())}.png",
                    mime="image/png"
                )
            
            # Reset button
            if st.button("Reset Statistics"):
                st.session_state.frames_processed = 0
                st.session_state.total_detections = 0
                st.session_state.detection_summary = {}
                st.rerun()
    
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
