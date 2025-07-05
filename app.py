import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import io

# Load the trained model
model = load_model("digit_classifier_v2.h5")

# Streamlit page configuration
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🎨 Handwritten Digit Recognizer")
st.markdown("### ✏️ Draw a digit (0–9) below:")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Buttons
col1, col2 = st.columns([1, 1])

if col1.button("🔍 Predict"):
    if canvas_result.image_data is not None:
        # Convert to image
        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        confidence = 100 * np.max(prediction)

        st.markdown("### 🧠 What the model sees:")
        st.image(img.reshape(28, 28), width=150)
        st.success(f"✅ Predicted: **{pred_class}** ({confidence:.2f}%)")
    else:
        st.warning("Please draw a digit to predict.")

if col2.button("🧹 Clear"):
    st.experimental_rerun()  # Forces the app to refresh and clear canvas
