import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

@st.cache_resource
def load_model():
    # Adding a try-except block helps debug if the file is missing
    try:
        return tf.keras.models.load_model("brain_tumor_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("🧠 Brain Tumor Detection System")
st.markdown("---")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

def preprocess(image):
    image = image.convert("RGB") # Critical for channel consistency
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0) # Cleaner than .reshape()
    return image

if uploaded_file and model:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    with st.spinner("Analyzing scan..."):
        processed_image = preprocess(image)
        prediction = model.predict(processed_image)
        
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    with col2:
        st.subheader("Results")
        if class_index == 0:
            st.error(f"⚠ Brain Tumor Detected")
            st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            st.success(f"✅ No Brain Tumor Detected")
            st.write(f"**Confidence:** {confidence:.2f}%")
