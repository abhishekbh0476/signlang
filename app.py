import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition System")
st.write("Upload a hand gesture image to predict the ASL alphabet (A–Z).")

@st.cache_resource
def load_cnn_model():
    return load_model("modelslr.h5")

model = load_cnn_model()

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader(
    "📤 Upload a hand gesture image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_letter = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"### ✅ Predicted Sign: **{predicted_letter}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

st.markdown("---")
st.markdown(
    """
    ### 📌 Notes
    - Use **clear hand gesture images**
    - White or plain background improves accuracy
    - Model supports **A–Z alphabets only**
    """
)
