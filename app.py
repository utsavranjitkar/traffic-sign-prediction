import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Traffic Sign Classifier",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #FDF4DC;
    color : #5B7B81;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/traffic_model.h5"

CLASS_NAMES = [
    "Construction work",
    "No entry",
    "Priority road",
    "Speed limit 50",
    "Stop",
    "Wild animal crossing"
]

IMG_SIZE = 224

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Traffic Sign Classification")
st.info("""
This model can predict the following traffic signs:
- Construction work
- No entry
- Priority road
- Speed limit 50
- Stop
- Wild animal crossing
""")

# TWO COLUMN LAYOUT
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=400)

with col2:
    if uploaded_file is not None:
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Prediction Result
            st.success(f"Predicted: {CLASS_NAMES[predicted_class]}")
            st.write(f"Confidence: {confidence*100:.2f}%")

            # Class Probabilities (UNDER the text)
            st.subheader("Class Probabilities")

            for i, prob in enumerate(prediction[0]):
                st.write(f"{CLASS_NAMES[i]} — {prob*100:.2f}%")
                st.progress(float(prob))