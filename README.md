
# 🧘‍♀️ AI Yoga Pose Detector

A real-time yoga pose classification app built with **Streamlit**, **OpenCV**, and **MediaPipe**, powered by a custom-trained machine learning model. Perfect for yoga enthusiasts, instructors, or anyone curious about their posture!

## 📸 Live Demo
Detect your yoga pose using your webcam and receive instant feedback with confidence scores and visual overlays.

## 🔍 Features
- 🎥 Real-time webcam feed
- 🧠 Pose prediction using a trained ML model
- 📊 Confidence score visualization
- 🖌️ Landmark drawing with MediaPipe
- 🧭 Streamlit sidebar controls

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [scikit-learn](https://scikit-learn.org/)
- Python 3.7+

## 📦 Installation

```bash
pip install streamlit opencv-python mediapipe numpy scikit-learn
```

Place your trained model file (`yoga_pose_model.pkl`) in the root directory.

## 🚀 Run the App

```bash
streamlit run app.py
```

## 🧠 Model Training
The model (`yoga_pose_model.pkl`) should be trained on pose landmark data extracted from MediaPipe. Each sample should include:
- 33 landmarks × (x, y, z, visibility) = 132 features
- Labeled yoga poses (e.g., Warrior, Tree, Cobra)

You can use classifiers like Random Forest, SVM, or XGBoost depending on your dataset.

## 📁 File Structure

```
├── app.py
├── yoga_pose_model.pkl
├── README.md
```

## 📸 Screenshot

> _Add a screenshot of the app in action here for visual appeal._

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## 📄 License
This project is open-source under the MIT License.

---
