import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- Load Model ---
try:
    with open("yoga_pose_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'yoga_pose_model.pkl' is in the same directory.")
    st.stop()

# --- Mediapipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- App UI ---
st.title("🧘‍♀️ AI Yoga Pose Detector")
st.markdown("This app uses your webcam to detect your yoga pose in real-time.")

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

# --- Camera Input ---
if run:
    img_file = st.camera_input("Turn on webcam")

    if img_file is not None:
        # Convert file to OpenCV image
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )
            try:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                row = [v for lm in landmarks for v in (lm.x, lm.y, lm.z, lm.visibility)]
                X = np.array(row).reshape(1, -1)

                pose_name = model.predict(X)[0]
                confidence = model.predict_proba(X)[0].max()

                status_placeholder.success(f"Detected: **{pose_name}**")
                confidence_bar.progress(float(confidence))
                confidence_text.markdown(f"**{confidence*100:.1f}%**")
            except Exception:
                status_placeholder.error("Error during prediction.")
        else:
            status_placeholder.warning("No person detected.")
            confidence_bar.progress(0)
            confidence_text.text("")

        # Show frame
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
