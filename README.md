# 🚦 Traffic Sign Prediction using Deep Learning

## 📌 Overview
This project is a deep learning–based traffic sign classification system that can identify different types of traffic signs from images. It uses **Transfer Learning with MobileNetV2** and is deployed as an interactive **Streamlit web application** for real-time predictions.

---

## 🎯 Features
- Classifies traffic signs into 6 categories:
  - Construction work
  - No entry
  - Priority road
  - Speed limit 50
  - Stop
  - Wild animal crossing
- Upload image and get instant prediction
- Displays prediction confidence
- Shows probability for each class
- Clean and user-friendly UI

---

## 🧠 Model Details
- **Model:** MobileNetV2 (Pretrained on ImageNet)
- **Approach:** Transfer Learning
- **Input Size:** 224 × 224
- **Output Layer:** Softmax (6 classes)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

---

## 🔄 Data Preprocessing & Augmentation
- Image resizing (224x224)
- Normalization (rescale 1./255)
- Data augmentation:
  - Rotation
  - Zoom
  - Horizontal flip

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- PIL

---

## 📂 Project Structure
