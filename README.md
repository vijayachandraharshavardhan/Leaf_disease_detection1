
# 🍃 Leaf Disease Detection using Deep Learning

A machine learning project focused on detecting plant leaf diseases from images to support early diagnosis and promote healthy crop management. This tool can assist farmers, researchers, and agri-tech professionals in identifying diseases with speed and accuracy.

---

## 🔍 Overview

This project uses Convolutional Neural Networks (CNNs) to classify various plant leaf diseases from image data. It includes image preprocessing, model training, real-time predictions, and an optional web interface for ease of use.

---

## ⚙️ Tech Stack

- **Python 3**
- **TensorFlow / Keras**
- **OpenCV** for image processing
- **NumPy, Pandas, Matplotlib** for data handling
- **Flask** for web application
- **(Legacy)** Streamlit application available in main.py

---

## 📂 Project Structure

```
leaf-disease-detection/
│
├── dataset/              # Leaf images (healthy and diseased)
├── Training/
│   └── model/            # Trained CNN model files
├── templates/            # HTML templates for Flask web app
│   ├── landing.html      # Welcome page
│   ├── dashboard.html    # Main dashboard
│   └── index.html        # Prediction page
├── API/                  # API related files
├── app.py                # Flask web application
├── main.py               # Streamlit application (legacy)
├── requirements.txt
├── train.py              # Model training script
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/leaf-disease-detection.git
cd leaf-disease-detection
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

### 4. Make predictions
```bash
python predict.py --image path_to_leaf_image.jpg
```

### 5. Run the web application
```bash
python app.py
```

The web application will start on `http://127.0.0.1:5000` with the following pages:
- **Landing Page** (`/`): Welcome page with navigation to dashboard
- **Dashboard** (`/dashboard`): Choose between camera scanning or file upload
- **Prediction** (`/predict`): Upload leaf images for disease detection

---

## 🧠 Model Highlights

- **Model Type:** CNN (custom or transfer learning)
- **Dataset:** Open-source PlantVillage or custom
- **Classes:** Healthy + various disease types
- **Accuracy:** ~90% (based on test data)

---

## 🌿 Sample Prediction

| Input Image                      | Output |
|----------------------------------|--------|
| `tomato_leaf_curl.jpg`           | Tomato Leaf Curl Virus |
| `potato_early_blight.jpg`        | Potato Early Blight     |

---

## 📈 Applications

- Smart farming and automated disease detection
- Mobile agriculture diagnostics
- Research and education in plant pathology

---

## 📘 License

This project is open-source under the MIT License. Feel free to use, modify, and share!

---


---
