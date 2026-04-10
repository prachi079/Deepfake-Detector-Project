import streamlit as st
from PIL import Image
from model_logic import predict_frame

st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.title("🛡️ Deepfake Detection System")
st.write("Upload an image to verify its authenticity using our AI model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("### Analysis Result")
        with st.spinner("Scanning pixels and faces..."):
            # Result aur confidence score lena
            result, confidence = predict_frame(frame)
            
            # Agar face nahi mila toh special message dikhayein
            if result == "No Face Detected":
                st.warning("⚠️ No Face Detected! Please upload a clear photo of a person.")
            else:
                if result == "Real":
                    st.success(f"✅ Result: {result}")
                    st.info(f"Confidence: {confidence}%")
                else:
                    st.error(f"🚨 Result: {result}")
                    st.info(f"Confidence: {confidence}%")

st.markdown("---")
st.caption("Note: This is an AI-based detection. For better results, use high-quality face images.")
