import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown("<h1 style='text-align: center;'>Skin Disease Classifier</h1>", unsafe_allow_html=True)

# Layout: Left (Controls + Treatment Info) | Right (Image + Results)
left_col, right_col = st.columns([1, 1.8])

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

# Treatment Info Dictionary
treatment_info = {
    "BA- cellulitis": "Treatment typically involves antibiotics. Seek medical attention promptly.",
    "BA-impetigo": "Usually treated with topical or oral antibiotics, depending on severity.",
    "FU-athlete-foot": "Antifungal creams, keeping feet dry, and proper footwear are recommended.",
    "FU-nail-fungus": "Treatment options include oral antifungal medications and medicated nail creams.",
    "FU-ringworm": "Topical antifungal medication is typically effective for treatment.",
    "PA-cutaneous-larva-migrans": "Treated with antiparasitic medications like albendazole or ivermectin.",
    "VI-chickenpox": "Treatment focuses on relieving symptoms; antiviral medications may be prescribed.",
    "VI-shingles": "Early treatment with antivirals can reduce the severity and duration."
}

# LEFT COLUMN - Upload & Treatment Info
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

    show_treatment = False
    predicted_class = None

    if uploaded_file is not None and not model_error:
        show_treatment = st.button("Get Treatment Information")

        # Convert image for prediction
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = cv2.resize(img_array, (128, 128))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        # Show treatment info directly below the button
        if show_treatment:
            st.markdown(f"""
            <div class="info-panel">
                <h4>Treatment Information:</h4>
                <p>{treatment_info.get(predicted_class, "No specific treatment available.")}</p>
                <div class="disclaimer">
                    <strong>Please consult a dermatologist for proper diagnosis and treatment.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# RIGHT COLUMN - Show Image + Predictions
with right_col:
    if uploaded_file is not None and not model_error:
        # **Show Image at the Top**
        st.image(image, caption=f"Uploaded Image", use_column_width=True)

        # **Show Predictions**
        st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
        st.markdown(f"<div class='disease-name'>Diagnosis: {predicted_class}</div>", unsafe_allow_html=True)

        # Top 3 Predictions
        st.markdown("<h4>Top Predictions:</h4>", unsafe_allow_html=True)
        top_indices = np.argsort(prediction[0])[-3:][::-1]

        for idx in top_indices:
            prob = float(prediction[0][idx]) * 100
            st.markdown(f"<div class='probability-item'><span>{class_labels[idx]}</span><span>{prob:.1f}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='probability-bar' style='width: {prob}%;'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-disclaimer">
    Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.
</div>
""", unsafe_allow_html=True)
