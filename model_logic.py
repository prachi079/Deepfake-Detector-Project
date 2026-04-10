import cv2
import numpy as np
import tensorflow as tf
from PIL import Image Model load karne ka function (agar pehle se nahi hai)
def predict_frame(uploaded_file):
    try:
        # 1. Image ko aise format mein badalna jo OpenCV samajh sake
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            return "Invalid Image", 0

        # 2. Image ko resize karna (Jo error aa raha tha wo isi line par tha)
        img = cv2.resize(image, (128, 128))
        img = img / 255.0  # Normalization
        img = np.expand_dims(img, axis=0)

        # 3. Yahan aapka model prediction karega
        # (Maanti hoon aapne model pehle hi load kiya hua hai)
        # result = model.predict(img) 
        
        return "Processing Done", 0.95 # Ye abhi testing ke liye hai

    except Exception as e:
        return f"Error: {str(e)}", 0
