import streamlit as st
import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image processing
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model.h5")


# Class labels (Ensure correct order)
class_labels = [
    "BA- cellulitis", "BA-impetigo", "FU-athlete-foot", "FU-nail-fungus",
    "FU-ringworm", "PA-cutaneous-larva-migrans", "VI-chickenpox", "VI-shingles"
]

st.title("Skin Disease Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image with PIL and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convert to NumPy array (BGR format, same as OpenCV)
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR
    img_array = cv2.resize(img_array, (128, 128))  # Resize to model input size
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Debugging: Display processed image shape
    st.write(f"Image Shape: {img_array.shape}")

    # Predict using model
    prediction = model.predict(img_array)

    # Debugging: Show raw output
    st.write(f"Raw Prediction Probabilities: {prediction}")

    # Find predicted class
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    # Show final prediction
    st.write(f"Predicted Class Index: {predicted_class_index}")
    st.write(f"Final Prediction: **{predicted_class}**")

    # **Show uploaded image with prediction title**
    st.image(image, caption=f"Prediction: {predicted_class}", use_column_width=True)
