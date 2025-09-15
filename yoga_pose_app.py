import streamlit as st
import cv2
import numpy as np
import pickle
from ultralytics import YOLO

# --- Load Yoga Pose Classifier ---
try:
    with open("yoga_pose_model.pkl", "rb") as f:
        clf = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'yoga_pose_model.pkl' is in the same directory.")
    st.stop()

# --- Load YOLOv8 Pose model ---
pose_model = YOLO("yolov8n-pose.pt")   # download auto on first run

# --- App UI ---
st.title("🧘‍♀️ AI Yoga Pose Detector (YOLOv8)")
st.markdown("This app uses YOLOv8 pose detection to classify yoga poses.")

# --- Sidebar ---
st.sidebar.header("Controls")
run = st.sidebar.checkbox("Start Webcam", value=True)
st.sidebar.markdown("---")
st.sidebar.header("Detection Status")
status_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.header("Confidence")
confidence_bar = st.sidebar.progress(0)
confidence_text = st.sidebar.empty()

# --- Main ---
st.header("Your Webcam Feed")
frame_placeholder = st.empty()

if run:
    img_file = st.camera_input("Turn on webcam")

    if img_file is not None:
        # Convert file to OpenCV image
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Run YOLOv8 pose detection
        results = pose_model.predict(frame, verbose=False)

        if results and results[0].keypoints is not None:
            # Keypoints shape: (num_people, num_keypoints, 3)
            kpts = results[0].keypoints.xy.cpu().numpy()
            if len(kpts) > 0:
                person = kpts[0]  # take first person
                # Flatten keypoints into row (x,y)
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
        frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
