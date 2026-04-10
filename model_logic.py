import cv2
import numpy as np
import tensorflow as tf
import os

# Model ko load karne ka sahi rasta
MODEL_PATH = "model.h5" # Pakka karein ki aapki model file ka naam yahi hai

def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_my_model()

def predict_frame(uploaded_file):
    try:
        # 1. Image ko read aur convert karna
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            return "Invalid Image", 0

        # 2. Preprocessing
        img = cv2.resize(image, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # 3. Prediction
        if model is not None:
            prediction = model.predict(img)
            confidence = float(prediction[0][0])
            # Agar 0.5 se zyada hai toh Fake, warna Real (apne model ke hisab se check karein)
            result = "Fake" if confidence > 0.5 else "Real"
            return result, confidence
        else:
            return "Model Not Found", 0

    except Exception as e:
        return f"Error: {str(e)}", 0
