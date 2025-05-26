import streamlit as st
from PIL import Image
import numpy as np
import joblib
from deepface import DeepFace

# Load your trained MLP model once
model = joblib.load("best_model.joblib")

# Page setup
st.set_page_config(
    page_title="Face-to-BMI Predictor",
    page_icon="💻",
    layout="centered"
)

# CSS for layout and style
st.markdown("""
    <style>
        .block-container {
            padding-top: 4rem;
            padding-bottom: 4rem;
            max-width: 1000px;   /* increased width for desktop */
            margin-left: auto;
            margin-right: auto;
        }
        h1.title {
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
            font-size: 3.2rem;    /* slightly bigger title */
            color: #1e2b5c;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.3rem;    /* slightly bigger subtitle */
            color: #555;
            margin-bottom: 3rem;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #999;
            margin-top: 4rem;
            margin-bottom: 4rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="title">Face-to-BMI Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a face image to estimate BMI using deep learning</div>', unsafe_allow_html=True)

# Upload image widget
uploaded_file = st.file_uploader("📸 Upload a face image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    st.markdown("🎯 **Predicting BMI...**")

    with st.spinner("Analyzing facial features with DeepFace..."):
        try:
            # Get embedding from the uploaded image
            embedding_obj = DeepFace.represent(img_path=np.array(image), model_name="VGG-Face", enforce_detection=False)
            embedding = embedding_obj[0]["embedding"]
            
            # Convert to numpy array and reshape for model
            embedding_np = np.array(embedding).reshape(1, -1)

            # Predict BMI using the loaded model
            bmi_pred = model.predict(embedding_np)[0]
            bmi_pred_rounded = round(float(bmi_pred), 1)

            # Categorize BMI
            if bmi_pred_rounded < 18.5:
                category = "Underweight"
                emoji = "🔵"
            elif bmi_pred_rounded < 25:
                category = "Normal"
                emoji = "🟢"
            elif bmi_pred_rounded < 30:
                category = "Overweight"
                emoji = "🟡"
            else:
                category = "Obese"
                emoji = "🔴"

            st.success(f"✅ Predicted BMI: **{bmi_pred_rounded}**  —  {emoji} **{category}**")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

# Footer
st.markdown("""
    <div class="footer">
        <hr style="margin-top: 3rem; margin-bottom: 1.5rem; border: none; height: 1px; background-color: #ccc;" />
        🧠 Created for the Machine Learning 2 Project<br>
        © 2025 MSADS at The University of Chicago
    </div>
""", unsafe_allow_html=True)
