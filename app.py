# Streamlit PPE Detection — Live Webcam (webrtc) Example

This single document contains everything you need to run a *true live webcam* PPE detection app in Streamlit using **streamlit-webrtc** and **Ultralytics YOLO**. Put `best.pt` in the same repo root (or change the path in the code).

---

## File: `app.py`

```python
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Page config
st.set_page_config(page_title="PPE Detection (WebRTC)", page_icon="🛡️", layout="wide")

# RTC config (public STUN). Modify if you need a TURN server.
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

@st.cache_resource
def load_model(path: str = "best.pt"):
    """Load YOLO model once and cache it."""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Failed loading model: {e}")
        return None

# Utility functions (same logic as your original code)
def filter_highest_confidence_per_class(results):
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
    img_copy = image.copy()

    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Draw box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{class_names[class_id]}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 8, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img_copy


# Video transformer for streamlit-webrtc
class PPETransformer(VideoTransformerBase):
    def __init__(self, model_path="best.pt", conf_thresh=0.4, iou_thresh=0.45):
        # Load the shared/cached model
        self.model = load_model(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def transform(self, frame):
        # frame is an av.VideoFrame; convert to BGR numpy array
        img = frame.to_ndarray(format="bgr24")

        if self.model is None:
            return img

        # Run inference
        try:
            results = self.model(img, conf=self.conf_thresh, iou=self.iou_thresh)
        except Exception as e:
            # If model inference fails, return original frame
            print("Inference error:", e)
            return img

        filtered = filter_highest_confidence_per_class(results[0])
        annotated = plot_filtered_results(img, filtered, self.model.names)

        return annotated


def main():
    st.title("PPE Detection — Real-time Webcam (WebRTC)")

    st.sidebar.header("Settings")
    conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.4)
    iou_thresh = st.sidebar.slider("IoU threshold", 0.1, 0.95, 0.45)
    model_path = st.sidebar.text_input("Model path", value="best.pt")

    st.sidebar.markdown("\nMake sure `best.pt` exists in the repo root or provide a path accessible to the app.")

    st.write("Click **Start** to begin real-time detection using your webcam. This uses WebRTC and sends webcam frames to the server for processing.")

    # Start WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="ppe-webrtc",
        mode="ACTIVE",
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=lambda: PPETransformer(model_path=model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    if webrtc_ctx.state.playing:
        st.success("Webcam stream started. Detection running...")

    # Optionally allow capturing a snapshot and downloading
    if st.button("Capture Snapshot"):
        # Capture a single frame from the transformer (if available)
        if webrtc_ctx.video_transformer is not None and hasattr(webrtc_ctx, "receiver"):
            # Grab the last frame by requesting a frame (webrtc internals)
            try:
                frame = webrtc_ctx.video_transformer.last_frame
            except Exception:
                frame = None

        # Fallback: ask user to take a picture using camera_input
        if 'frame' in locals() and frame is not None:
            img = frame
            # img expected as numpy BGR
            _, im_buf_arr = cv2.imencode('.png', img)
            st.download_button("Download snapshot", data=im_buf_arr.tobytes(), file_name="ppe_snapshot.png", mime="image/png")
        else:
            st.info("Snapshot not available. Use the camera_input mode as a fallback to take a still photo.")


if __name__ == '__main__':
    main()

