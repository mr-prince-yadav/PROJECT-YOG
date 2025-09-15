import streamlit as st
import numpy as np
import pickle
from PIL import Image
import mediapipe as mp

# Load Yoga Pose Classifier
try:
    with open("yoga_pose_model.pkl", "rb") as f:
        clf = pickle.load(f)
except FileNotFoundError:
    st.error("Place 'yoga_pose_model.pkl' in the app directory.")
    st.stop()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("AI Yoga Pose Detector (Mediapipe)")
st.markdown("Detect yoga poses using Mediapipe Pose.")

# Sidebar controls
st.sidebar.header("Controls")
run = st.sidebar.checkbox("Start Webcam", value=True)
st.sidebar.header("Detection Status")
status_placeholder = st.sidebar.empty()
st.sidebar.header("Confidence")
confidence_bar = st.sidebar.progress(0)
confidence_text = st.sidebar.empty()

st.header("Your Webcam Feed")
frame_placeholder = st.empty()

if run:
    img_file = st.camera_input("Turn on webcam")
    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)

        # Mediapipe Pose Detection
        results = pose.process(frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            X = np.array(row).reshape(1, -1)

            # Predict pose
            pose_name = clf.predict(X)[0]
            confidence = clf.predict_proba(X)[0].max()
            status_placeholder.success(f"Detected: **{pose_name}**")
            confidence_bar.progress(float(confidence))
            confidence_text.markdown(f"**{confidence*100:.1f}%**")

            # Annotate frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            status_placeholder.warning("No person detected.")
            confidence_bar.progress(0)
            confidence_text.text("")

        frame_placeholder.image(frame, channels="RGB", use_container_width=True)
