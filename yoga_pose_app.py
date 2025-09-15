import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- Load Model and Initialize MediaPipe ---
try:
    with open('yoga_pose_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'yoga_pose_model.pkl' is in the same directory.")
    st.stop()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- App UI ---
st.title("🧘‍♀️ AI Yoga Pose Detector")
st.markdown("This app uses your webcam to detect your yoga pose in real-time.")

# --- Sidebar ---
st.sidebar.header("Controls")
run = st.sidebar.checkbox('Start Webcam', value=True)
st.sidebar.markdown("---")
st.sidebar.header("Detection Status")
status_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.header("Confidence")
confidence_bar = st.sidebar.progress(0)
confidence_text = st.sidebar.empty()

# --- Main Layout (Single Column) ---
st.header("Your Webcam Feed")
frame_placeholder = st.empty()

# --- Webcam Loop ---
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam feed ended.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        try:
            # Extract landmarks and make prediction
            landmarks = results.pose_landmarks.landmark
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            X = np.array(row).reshape(1, -1)
            
            pose_name = model.predict(X)[0]
            confidence = model.predict_proba(X)[0].max()

            # Update sidebar UI
            status_placeholder.success(f"Detected: **{pose_name}**")
            confidence_bar.progress(float(confidence))
            confidence_text.markdown(f"**{confidence*100:.1f}%**")

        except Exception as e:
            status_placeholder.error("Error during prediction.")
    else:
        # If no landmarks are detected, reset the status
        status_placeholder.warning("No person detected.")
        confidence_bar.progress(0)
        confidence_text.text("")

    # Display the webcam feed
    frame_placeholder.image(frame, channels="BGR", use_container_width=True)

cap.release()
cv2.destroyAllWindows()

st.success("Webcam has been turned off.")