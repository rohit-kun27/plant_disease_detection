import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# -----------------------------------------------------------
# Load the trained model
# -----------------------------------------------------------
model = load_model("vgg19_plant_disease_final.h5")

# -----------------------------------------------------------
# Class Labels (update if needed to match your dataset)
# -----------------------------------------------------------
class_labels = sorted([
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Tomato_Bacterial_spot",
"Tomato_Early_blight",
"Tomato_Late_blight",
"Tomato_Leaf_Mold",
"Tomato_Septoria_leaf_spot",
"Tomato_Spider_mites_Two_spotted_spider_mite",
"Tomato__Target_Spot",
"Tomato__Tomato_YellowLeaf__Curl_Virus",
"Tomato__Tomato_mosaic_virus",
"Tomato_healthy"
])

# -----------------------------------------------------------
# Streamlit App UI
# -----------------------------------------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection (VGG19)")
st.write("Upload a plant leaf image to detect the disease.")

uploaded_file = st.file_uploader("📷 Choose a leaf image...", type=["jpg", "png", "jpeg"])

# -----------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing...")

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display result
    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.info(f"🎯 Confidence: {confidence:.2f}%")
