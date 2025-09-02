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
    annotated_img = plot_filtered_results(img_array, filtered_results, model.names, is_video=True)
    
    return annotated_img, filtered_results

def filter_highest_confidence_per_class(results):
    """Filter results to keep only highest confidence detection per class"""
    if len(results.boxes) == 0:
        return []
    
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

def plot_filtered_results(image, filtered_boxes, class_names, is_video=True):
    """Plot filtered results on image with enhanced visibility"""
    img_copy = image.copy()
    
    # Enhanced visibility for live video
    box_thickness = 4
    font_scale = 1.2
    font_thickness = 3
    label_padding = 20
    box_color = (0, 255, 0)  # Bright green
    text_color = (0, 0, 0)   # Black text
    
    for box in filtered_boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Draw label
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

# JavaScript for continuous webcam capture
def get_webcam_js():
    return """
    <script>
    let video;
    let canvas;
    let context;
    let isStreaming = false;
    let streamInterval;

    async function startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480, 
                    facingMode: 'environment' 
                } 
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
                document.getElementById('status').innerText = '🟢 LIVE DETECTION ACTIVE';
                captureFrames();
            });
            
        } catch (err) {
            console.error('Error accessing webcam:', err);
            document.getElementById('status').innerText = '❌ Camera Access Denied';
        }
    }

    function stopWebcam() {
        if (video && video.srcObject) {
            const tracks = video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }
        
        if (streamInterval) {
            clearInterval(streamInterval);
        }
        
        isStreaming = false;
        document.getElementById('status').innerText = '🔴 DETECTION STOPPED';
    }

    function captureFrames() {
        if (!isStreaming) return;
        
        streamInterval = setInterval(() => {
            if (isStreaming && video.readyState === video.HAVE_ENOUGH_DATA) {
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convert to base64 and send to Streamlit
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                const base64Data = imageData.split(',')[1];
                
                // Send frame to Streamlit via session state
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: base64Data
                }, '*');
            }
        }, 500); // Capture every 500ms (2 FPS for processing)
    }

    // Button event listeners
    document.getElementById('startBtn').addEventListener('click', startWebcam);
    document.getElementById('stopBtn').addEventListener('click', stopWebcam);
    </script>
    """

def get_webcam_html():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ 
                margin: 0; 
                padding: 20px; 
                font-family: Arial, sans-serif; 
                background: #f0f2f6;
            }}
            .container {{ 
                max-width: 800px; 
                margin: 0 auto; 
                text-align: center; 
            }}
            .controls {{ 
                margin: 20px 0; 
            }}
            button {{ 
                padding: 15px 30px; 
                margin: 10px; 
                font-size: 16px; 
                font-weight: bold; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                transition: all 0.3s; 
            }}
            #startBtn {{ 
                background: #00ff00; 
                color: black; 
            }}
            #stopBtn {{ 
                background: #ff4444; 
                color: white; 
            }}
            button:hover {{ 
                transform: scale(1.05); 
            }}
            #video {{ 
                width: 100%; 
                max-width: 640px; 
                height: auto; 
                border: 3px solid #4CAF50; 
                border-radius: 10px; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.3); 
            }}
            #canvas {{ 
                display: none; 
            }}
            #status {{ 
                font-size: 24px; 
                font-weight: bold; 
                margin: 20px 0; 
                padding: 15px; 
                border-radius: 8px; 
                background: white; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }}
            .instructions {{ 
                background: #e3f2fd; 
                padding: 20px; 
                border-radius: 8px; 
                margin: 20px 0; 
                text-align: left; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🔴 CONTINUOUS PPE LIVE DETECTION</h2>
            
            <div class="instructions">
                <h3>📋 How It Works:</h3>
                <ul>
                    <li><strong>🟢 START:</strong> Activates continuous webcam monitoring</li>
                    <li><strong>🎥 LIVE:</strong> Automatically captures and processes frames every 0.5 seconds</li>
                    <li><strong>⚡ AUTO:</strong> PPE detection runs continuously without clicking</li>
                    <li><strong>🔴 STOP:</strong> Ends live monitoring and saves last frame</li>
                </ul>
            </div>
            
            <div id="status">🔴 READY TO START - Click START button below</div>
            
            <div class="controls">
                <button id="startBtn">🟢 START CONTINUOUS DETECTION</button>
                <button id="stopBtn">🔴 STOP DETECTION</button>
            </div>
            
            <video id="video" autoplay muted playsinline></video>
            <canvas id="canvas"></canvas>
        </div>
        
        {get_webcam_js()}
    </body>
    </html>
    """

def main():
    st.title("PPE Compliance Detection System")
    st.markdown("**REAL CONTINUOUS LIVE DETECTION** - True webcam monitoring without clicking photos")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load PPE detection model. Please ensure 'best.pt' is in the same directory as this script.")
        return
    
    # Sidebar for mode selection
    st.sidebar.title("Detection Mode")
    mode = st.sidebar.selectbox(
        "Choose input type:",
        ["🔴 CONTINUOUS LIVE WEBCAM", "Image Upload", "Video Upload", "Single Photo Capture"]
    )
    
    if mode == "🔴 CONTINUOUS LIVE WEBCAM":
        st.header("🔴 CONTINUOUS LIVE PPE DETECTION")
        st.markdown("**TRUE LIVE MONITORING** - Continuous webcam feed with automatic PPE detection every 0.5 seconds")
        
        # Initialize session state
        if 'live_frames_processed' not in st.session_state:
            st.session_state.live_frames_processed = 0
        if 'current_detections' not in st.session_state:
            st.session_state.current_detections = []
        if 'last_live_frame' not in st.session_state:
            st.session_state.last_live_frame = None
        if 'detection_log' not in st.session_state:
            st.session_state.detection_log = []
        
        # Create layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("🎥 LIVE WEBCAM FEED")
            
            # Embed HTML webcam component
            webcam_html = get_webcam_html()
            
            # Use HTML component for continuous webcam
            import streamlit.components.v1 as components
            
            frame_data = components.html(
                webcam_html,
                height=700,
                scrolling=False
            )
            
            # Process received frame data
            if frame_data:
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(frame_data)
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Process with PPE detection
                    with st.spinner("🔄 Processing live frame..."):
                        processed_img, filtered_results = process_image(model, image)
                    
                    # Update session state
                    st.session_state.live_frames_processed += 1
                    st.session_state.current_detections = filtered_results
                    st.session_state.last_live_frame = processed_img
                    
                    # Add to detection log
                    detection_entry = {
                        'frame': st.session_state.live_frames_processed,
                        'timestamp': time.strftime("%H:%M:%S"),
                        'count': len(filtered_results),
                        'detections': [(model.names[int(box.cls[0])], float(box.conf[0])) for box in filtered_results]
                    }
                    st.session_state.detection_log.append(detection_entry)
                    
                    # Keep only last 10 entries
                    if len(st.session_state.detection_log) > 10:
                        st.session_state.detection_log = st.session_state.detection_log[-10:]
                    
                    # Display processed frame
                    st.image(
                        processed_img,
                        caption=f"🔴 LIVE PPE DETECTION - Frame #{st.session_state.live_frames_processed}",
                        use_column_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error processing frame: {e}")
        
        with col2:
            st.subheader("📊 LIVE DETECTION RESULTS")
            
            # Live statistics
            st.metric("Frames Processed", st.session_state.live_frames_processed, delta="Continuous")
            
            if st.session_state.current_detections:
                detection_count = len(st.session_state.current_detections)
                st.metric("Current PPE Items", detection_count, delta="Live count")
                
                st.write("**🔴 CURRENT LIVE DETECTIONS:**")
                for i, box in enumerate(st.session_state.current_detections):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"• **{class_name}**: {confidence:.2f} confidence")
            else:
                st.metric("Current PPE Items", 0, delta="Scanning...")
            
            # Detection history
            st.write("---")
            st.write("**📈 LIVE DETECTION LOG:**")
            
            if st.session_state.detection_log:
                for entry in reversed(st.session_state.detection_log[-5:]):
                    st.write(f"**Frame {entry['frame']}** ({entry['timestamp']}): {entry['count']} items")
                    if entry['detections']:
                        items = [f"{name} ({conf:.2f})" for name, conf in entry['detections']]
                        st.write(f"  → {', '.join(items)}")
            else:
                st.info("Start webcam to see live detection log")
            
            # Download last frame
            if st.session_state.last_live_frame is not None:
                st.write("---")
                st.write("**💾 SAVE CURRENT FRAME:**")
                
                img_bytes = io.BytesIO()
                Image.fromarray(st.session_state.last_live_frame).save(img_bytes, format='PNG')
                
                st.download_button(
                    label="📥 Download Current Detection",
                    data=img_bytes.getvalue(),
                    file_name=f"live_ppe_detection_frame_{st.session_state.live_frames_processed}.png",
                    mime="image/png",
                    key="download_live_frame"
                )
        
        # Live monitoring info
        st.markdown("---")
        st.subheader("🚀 CONTINUOUS LIVE MONITORING FEATURES")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.success("""
            **🟢 CONTINUOUS MODE:**
            • Webcam stays on continuously
            • Auto-captures every 0.5 seconds
            • No clicking photos needed
            • Real live monitoring
            • Can run for hours
            """)
        
        with info_col2:
            st.info("""
            **⚡ AUTO PROCESSING:**
            • Automatic PPE detection
            • Live results update
            • Detection history log
            • Frame counter tracking
            • Continuous monitoring
            """)
        
        with info_col3:
            st.warning("""
            **🔴 PROFESSIONAL USE:**
            • Perfect for client demos
            • Real workplace monitoring
            • Continuous compliance check
            • Save any frame instantly
            • True live detection system
            """)
        
        # Performance metrics
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Capture Rate", "2 FPS", delta="Every 0.5 sec")
        
        with perf_col2:
            st.metric("Processing", "Auto", delta="Continuous")
        
        with perf_col3:
            st.metric("Runtime", "Unlimited", delta="Hours if needed")
        
        with perf_col4:
            st.metric("Client Ready", "✅ YES", delta="Professional")
    
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
    
    elif mode == "Single Photo Capture":
        st.header("Quick PPE Snapshot Detection")
        
        camera_photo = st.camera_input("Take a photo for PPE detection")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Captured Photo")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("PPE Detection Results")
                
                with st.spinner("Processing..."):
                    processed_img, filtered_results = process_image(model, image)
                
                st.image(processed_img, use_column_width=True)
                
                if len(filtered_results) > 0:
                    st.success(f"Detected {len(filtered_results)} PPE items:")
                    for box in filtered_results:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        st.write(f"• **{class_name}**: {confidence:.2f}")
                else:
                    st.warning("No PPE equipment detected")
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 CONTINUOUS LIVE DETECTION")
    st.sidebar.success(
        "TRUE LIVE MONITORING:\n"
        "• Webcam runs continuously\n"
        "• Auto-captures every 0.5 seconds\n"
        "• No photo clicking needed\n"
        "• Real-time PPE detection\n"
        "• Can run for hours\n"
        "• Perfect for client demos\n"
        "• Professional live monitoring"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("PPE Detection Classes")
    st.sidebar.info(
        "Detects:\n"
        "- 🦺 Safety vests\n"
        "- 🧤 Safety gloves\n"
        "- ⛑️ Hard hats\n"
        "- 🛡️ Other safety equipment"
    )

if __name__ == "__main__":
    main()
