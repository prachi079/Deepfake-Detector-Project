import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2 

model_path = 'deepfake_model_v3.h5'
model = load_model(model_path) if os.path.exists(model_path) else None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_image(pil_img):
    if model is None: return "Model Error", 0

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if len(faces) == 0: return "No Face Detected", 0

    x, y, w, h = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    
    # 1. Texture Check (Laplacian Variance)
    # Real photos usually have variance between 100-500. 
    # AI/Smooth photos are usually below 60.
    laplacian_var = cv2.Laplacian(cropped_face, cv2.CV_64F).var()
    
    # 2. Model Score
    face_resized = cv2.resize(cropped_face, (299, 299))
    img_array = image.img_to_array(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    # --- THE GOLDEN BALANCE LOGIC ---
    # Agar model keh raha hai Real (score > 0.5) AUR photo grainy hai (var > 70) -> REAL
    if score > 0.50 and laplacian_var > 70:
        return "Real", round(score * 100, 2)
    
    # Agar dono mein se koi bhi ek cheez shaq paida kare -> DEEPFAKE
    else:
        # Final safety check: Agar bohot hi clear photo hai tabhi Real bolna
        if score > 0.85: 
            return "Real", round(score * 100, 2)
        else:
            return "Deepfake", round((1 - score) * 100, 2)