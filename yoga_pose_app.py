import streamlit as st
import numpy as np
import pickle
from PIL import Image
from ultralytics import YOLO

# --- Load Yoga Pose Classifier ---
try:
    with open("yoga_pose_model.pkl", "rb") as f:
        clf = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Place 'yoga_pose_model.pkl' in the app directory.")
    st.stop()

# --- Load YOLOv8 Pose Model ---
pose_model = YOLO("yolov8n-pose.pt")  # Auto-download if not present

# --- Streamlit UI ---
st.title("AI Yoga Pose Detector (YOLOv8)")
st.markdown("Detect yoga poses using YOLOv8 pose estimation.")

# Sidebar controls
st.sidebar.header("Controls")
run = st.sidebar.checkbox("Start Webcam", value=True)
st.sidebar.markdown("---")
st.sidebar.header("Detection Status")
status_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.header("Confidence")
confidence_bar = st.sidebar.progress(0)
confidence_text = st.sidebar.empty()

# Main webcam feed
st.header("Your Webcam Feed")
frame_placeholder = st.empty()

if run:
    img_file = st.camera_input("Turn on webcam")

    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)  # Convert PIL to numpy array (RGB)

        # Run YOLOv8 pose detection
        results = pose_model.predict(frame, verbose=False)

        if results and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy.cpu().numpy()
            if len(kpts) > 0:
                person = kpts[0]  # Take first detected person
                row = person.flatten()
                X = np.array(row).reshape(1, -1)

                # Predict yoga pose
                pose_name = clf.predict(X)[0]
                confidence = clf.predict_proba(X)[0].max()

                status_placeholder.success(f"Detected: **{pose_name}**")
                confidence_bar.progress(float(confidence))
                confidence_text.markdown(f"**{confidence*100:.1f}%**")
            else:
                status_placeholder.warning("No person detected.")
                confidence_bar.progress(0)
                confidence_text.text("")
        else:
            status_placeholder.warning("No person detected.")
            confidence_bar.progress(0)
            confidence_text.text("")

        # Annotated frame from YOLO
        annotated_frame = results[0].plot()
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
