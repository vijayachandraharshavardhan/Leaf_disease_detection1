import streamlit as st
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Sidebar with project info
st.sidebar.title("🌿 Leaf Disease Detector")
st.sidebar.markdown("""
**Detect plant leaf diseases using deep learning.**

- Supports 33 types of diseases
- Works for Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, Tomato

[GitHub](https://github.com/shukur-alom/leaf-diseases-detect)
""")

st.title("🌱 Leaf Disease Detection")
st.markdown("""
Welcome! This app uses a deep learning model to identify diseases in plant leaves from your images.

**How to use:**
1. Upload a clear photo of a single leaf (Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, or Tomato).
2. Wait for the model to analyze and predict the disease.
3. See the result and confidence below!

*Note: For best results, use images with only one leaf and a plain background.*
""")

# Load the model
try:
    model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')
except Exception as e:
    st.error(f"🚫 Failed to load model: {e}")
    model = None

label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

uploaded_file = st.file_uploader("📤 Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        if img is None:
            st.warning("⚠️ Could not decode image. Please upload a valid image file.")
        elif model is None:
            st.error("🚫 Model not loaded. Cannot make predictions.")
        else:
            with st.spinner('Analyzing image...'):
                normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
                predictions = model.predict(normalized_image)
            st.image(image_bytes, caption="Uploaded Leaf", use_container_width=True)
            confidence = predictions[0][np.argmax(predictions)]*100
            result_label = label_name[np.argmax(predictions)]
            if confidence >= 80:
                st.success(f"✅ **Prediction:** {result_label}\n**Confidence:** {confidence:.2f}%")
            else:
                st.warning("🤔 The model is not confident in its prediction. Please try another image.")
    except Exception as e:
        st.error(f"🚫 **Error processing image:** {e}")