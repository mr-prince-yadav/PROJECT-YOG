import streamlit as st
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image

# -------------------- Load Model --------------------
try:
    with open("yoga_pose_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'yoga_pose_model.pkl' not found.")
    st.stop()

# -------------------- MediaPipe Init --------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- UI --------------------
st.title("🧘 AI Yoga Pose Detector")
st.markdown("Detect yoga poses using your webcam (Streamlit Cloud safe)")

st.sidebar.header("Controls")
run = st.sidebar.checkbox("Start Webcam", True)

status = st.sidebar.empty()
confidence_bar = st.sidebar.progress(0)
confidence_text = st.sidebar.empty()

frame_placeholder = st.empty()

# -------------------- Webcam Input --------------------
if run:
    img_file = st.camera_input("Turn on webcam")

    if img_file:
        # Convert to PIL → NumPy (NO OpenCV)
        image = Image.open(img_file).convert("RGB")
        image_np = np.array(image)

        # MediaPipe expects RGB
        results = pose.process(image_np)

        if results.pose_landmarks:
            # Draw landmarks (MediaPipe uses matplotlib internally)
            annotated_image = image_np.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            try:
                landmarks = results.pose_landmarks.landmark
                row = [v for lm in landmarks for v in (lm.x, lm.y, lm.z, lm.visibility)]
                X = np.array(row).reshape(1, -1)

                pose_name = model.predict(X)[0]
                confidence = model.predict_proba(X)[0].max()

                status.success(f"Detected: **{pose_name}**")
                confidence_bar.progress(float(confidence))
                confidence_text.markdown(f"**{confidence*100:.1f}% confidence**")

            except Exception as e:
                status.error("Prediction error")

            frame_placeholder.image(
                annotated_image,
                channels="RGB",
                use_container_width=True
            )

        else:
            status.warning("No person detected")
            confidence_bar.progress(0)
            frame_placeholder.image(image, use_container_width=True)

else:
    st.info("Webcam stopped")
