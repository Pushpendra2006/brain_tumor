import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to detect Brain Tumor")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

def preprocess(image):
    image = image.resize((128,128))
    image = np.array(image)/255.0
    image = image.reshape(1,128,128,3)
    return image

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess(image)
    prediction = model.predict(processed_image)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)*100

    if class_index == 0:
        st.error(f"⚠  Brain Tumor Detected ({confidence:.2f}%)")
    else:
        st.success(f"✅ No Brain Tumor Detected ({confidence:.2f}%)")
