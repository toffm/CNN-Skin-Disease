import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load CSS for styling
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Page title
st.markdown("<h1 style='text-align: center;'>Skin Disease Classifier</h1>", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

try:
    model = load_model()
    model_error = False
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_error = True

# Class labels
class_labels = [
    "BA- cellulitis", "BA-impetigo", "FU-athlete-foot", "FU-nail-fungus",
    "FU-ringworm", "PA-cutaneous-larva-migrans", "VI-chickenpox", "VI-shingles"
]

# Layout columns
left_col, right_col = st.columns([1, 1.8])

with left_col:
    st.markdown("<h3>Upload an image for diagnosis</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "png", "jpeg"])

    st.markdown("""
    <div class="info-panel">
        <h4>Supported Conditions:</h4>
        <ul>
            <li>Bacterial: Cellulitis, Impetigo</li>
            <li>Fungal: Athlete's foot, Nail fungus, Ringworm</li>
            <li>Parasitic: Cutaneous larva migrans</li>
            <li>Viral: Chickenpox, Shingles</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None and not model_error:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = cv2.resize(img_array, (128, 128)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index]) * 100

        st.markdown(f"<div class='result-panel'>", unsafe_allow_html=True)
        st.markdown(f"<div class='disease-name'>Diagnosis: {predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)

        # Display top 3 predictions
        st.markdown("<h4>Top Predictions:</h4>", unsafe_allow_html=True)
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        for idx in top_indices:
            prob = float(prediction[0][idx]) * 100
            st.markdown(f"<div class='probability-item'><span>{class_labels[idx]}</span><span>{prob:.1f}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='probability-bar' style='width: {prob}%;'></div>", unsafe_allow_html=True)

        # Get Treatment Information Button
        if st.button("Get Treatment Information"):
            treatment_info = {
                "BA- cellulitis": "Antibiotics are commonly used. Keep the affected area clean.",
                "BA-impetigo": "Treated with topical or oral antibiotics.",
                "FU-athlete-foot": "Use antifungal creams and keep feet dry.",
                "FU-nail-fungus": "Medicated nail creams or oral antifungals may be needed.",
                "FU-ringworm": "Apply topical antifungal creams.",
                "PA-cutaneous-larva-migrans": "Antiparasitic medications like albendazole are effective.",
                "VI-chickenpox": "Rest and symptomatic relief; antivirals for severe cases.",
                "VI-shingles": "Antivirals can reduce symptoms if taken early."
            }
            st.markdown(f"""
            <div class="info-panel">
                <h4>Treatment Information:</h4>
                <p>{treatment_info.get(predicted_class, "Please consult a dermatologist for proper diagnosis and treatment.")}</p>
                <div class="disclaimer">
                    Please consult a dermatologist for proper diagnosis and treatment.
                </div>
            </div>
            """, unsafe_allow_html=True)

# Right column: Image Display
with right_col:
    if uploaded_file is not None:
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder-container">
            <div class="placeholder-icon">📷</div>
            <h3>Upload an image to get started</h3>
            <p>Supported formats: JPG, PNG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-disclaimer">
    Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.
</div>
""", unsafe_allow_html=True)
