import os
import gdown
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# --- Google Drive se Model Download karne ka Logic ---
file_id = '1BesK2Syr1KadOOlqabu2wPJeS-lKZlRy'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'deepfake_model_v3.h5'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

model = load_model(output)

def predict_frame(frame):
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return "Real" if prediction < 0.5 else "Deepfake/Filtered", prediction