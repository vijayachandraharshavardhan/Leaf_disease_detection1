import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, EarlyStopping

# Assuming the dataset is downloaded and extracted to 'dataset' folder
dataset_path = 'dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/'

n_of_image, label_name = 100, ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy',
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot',
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

img, label, img_size = [], [], (150, 150)

# List of subfolders for each class
folders = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

for idx, folder in enumerate(folders):
    path_dir = os.path.join(dataset_path, folder)
    if os.path.exists(path_dir):
        img_path_list = os.listdir(path_dir)
        for len_no, img_path in enumerate(img_path_list):
            if len_no == n_of_image:
                break
            try:
                img.append(img_to_array(load_img(os.path.join(path_dir, img_path), target_size=img_size)) / 255)
                label.append(idx)
            except:
                pass

img, label = np.array(img), np.array(label)
IMG_SHAPE = img_size + (3,)

vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
vgg.trainable = False

average_pool = tf.keras.layers.GlobalAveragePooling2D()

prediction = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(33, activation='softmax')
])

tl_model = tf.keras.Sequential([
    vgg,
    average_pool,
    prediction,
])

tl_model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

img_train, img_test, label_train, label_test = train_test_split(img, label, test_size=0.25, random_state=10)

class mycallback(Callback):
    def on_train_end(self, epoch, log={}):
        try:
            val_acc = log.get('val_accuracy')
            if val_acc is not None and val_acc >= 0.90:
                print('Reached 90% accuracy so cancelling training')
                self.model.stop_training = True
        except:
            pass

custom_call = mycallback()
early_stop = EarlyStopping(monitor='val_accuracy', patience=3)

try:
    tl_model.fit(img_train, to_categorical(label_train), epochs=5, validation_data=(img_test, to_categorical(label_test)), callbacks=[custom_call, early_stop])
except Exception as e:
    print(f"Training error: {e}")

tl_model.save('Training/model/Leaf Deases(96,88)')

print("Model saved successfully!")