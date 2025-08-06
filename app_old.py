import streamlit as st
from PIL import Image
from model_helper import load_model, predict_image

# Load model once
@st.cache_resource
def get_model():
    return load_model("saved_model.pth")

model = get_model()

# Streamlit UI
st.title("🚘 Car Damage Detection App")
st.write("Upload a car image and the model will classify the type and location of damage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Make prediction
    predicted_class, confidence = predict_image(image, model)

    # Display result
    st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2%}`")
