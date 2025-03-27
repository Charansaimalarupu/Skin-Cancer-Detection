import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

# Load model (HDF5)
@st.cache_resource
def load_model_h5():
    st.write("📦 Loading model...")
    return load_model("best_model_tf.h5", compile=False)

model = load_model_h5()

# Dummy metadata input
def get_static_metadata():
    return np.zeros((1, 72), dtype=np.float32)

# Preprocess uploaded image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# UI
st.title("🔬 Skin Cancer Detection (Multimodal AI)")
st.markdown("Upload a **skin lesion image** to classify it as **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="✅ Uploaded Image", use_column_width=True)
    image_array = preprocess_image(uploaded_file)
    tabular_array = get_static_metadata()

    with st.spinner("🔍 Making prediction..."):
        prediction = model.predict([image_array, tabular_array])[0][0]
        label = "🟢 Benign (Non-Cancer)" if prediction >= 0.5 else "🔴 Malignant (Cancer)"
        prob = prediction if prediction >= 0.5 else 1 - prediction

    st.subheader("🧪 Prediction Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** `{prob:.2%}`")

