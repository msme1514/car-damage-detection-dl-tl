import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model_helper import load_model, predict_image, class_names

@st.cache_resource
def get_model():
    return load_model("saved_model.pth")

model = get_model()

st.title("🚘 Car Damage Detection App")
st.write("Upload a car image to classify the type and location of damage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("⏳ Predicting...")

    predicted_class, confidence, probabilities, gradcam = predict_image(image, model)

    st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2%}`")

    # Show bar chart of all class probabilities
    st.subheader("📊 Class Probabilities")
    st.bar_chart({class_names[i]: probabilities[i] for i in range(len(class_names))})

    # Grad-CAM heatmap overlaid on original image
    st.subheader("🔥 Grad-CAM Heatmap")

    def overlay_heatmap(pil_img, cam):
        img_np = np.array(pil_img.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        return overlay

    heatmap_overlay = overlay_heatmap(image, gradcam)
    st.image(heatmap_overlay, caption="Grad-CAM Overlay", use_container_width=True)
