from flask import Flask, render_template, request, redirect, url_for, flash
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = None
try:
    model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')
except Exception as e:
    print(f"Failed to load model: {e}")

label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy',
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot',
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(image_bytes):
    """Process image and return prediction"""
    try:
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        if img is None:
            return None, "Could not decode image. Please upload a valid image file."

        if model is None:
            return None, "Model not loaded. Cannot make predictions."

        # Preprocess image
        normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
        predictions = model.predict(normalized_image)

        confidence = predictions[0][np.argmax(predictions)] * 100
        result_label = label_name[np.argmax(predictions)]

        # Convert image to base64 for display
        _, buffer = cv.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'label': result_label,
            'confidence': confidence,
            'image_data': img_base64,
            'high_confidence': confidence >= 80
        }, None

    except Exception as e:
        return None, f"Error processing image: {str(e)}"

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_bytes = file.read()

            result, error = process_image(image_bytes)

            if error:
                flash(error, 'error')
                return redirect(request.url)

            return render_template('index.html', result=result)

        else:
            flash('Invalid file type. Please upload JPG, JPEG, or PNG files.', 'error')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)