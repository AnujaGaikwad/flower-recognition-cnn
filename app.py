import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="AI Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Recognition")
st.markdown(
    "Upload a picture of a flower and the AI will identify it."
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_flower_model():
    return tf.keras.models.load_model("custom_flower_model.h5")

try:
    model = load_flower_model()
except OSError:
    st.error(
        "Model not found! Please run train_brain.py first to generate custom_flower_model.h5."
    )
    st.stop()

# ==========================================
# CLASS NAMES
# ==========================================
CLASS_NAMES = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

# ==========================================
# IMAGE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "Choose a flower image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("🧠 The AI is analyzing the image...")

    # ==========================================
    # PREPROCESS IMAGE
    # ==========================================
    img_resized = image.resize((150, 150))

    img_array = np.array(img_resized)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ==========================================
    # PREDICTION
    # ==========================================
    predictions = model.predict(img_array, verbose=0)[0]

    highest_confidence_index = np.argmax(predictions)

    predicted_flower = CLASS_NAMES[highest_confidence_index]

    confidence_score = predictions[highest_confidence_index] * 100

    # ==========================================
    # RESULTS
    # ==========================================
    st.markdown("---")

    if confidence_score > 70:
        st.success(f"🎯 Prediction: {predicted_flower}")
    else:
        st.warning(f"🤔 Best Guess: {predicted_flower}")
        st.caption(
            "The AI is not highly confident. Try another image or better lighting."
        )

    st.info(
        f"Confidence Score: {confidence_score:.2f}%"
    )

    # ==========================================
    # PROBABILITY CHART
    # ==========================================
    st.markdown("### AI Confidence Distribution")

    chart_data = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(chart_data)
