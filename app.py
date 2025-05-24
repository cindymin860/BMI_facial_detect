import streamlit as st
from PIL import Image
import numpy as np

# Page setup
st.set_page_config(
    page_title="Face-to-BMI Predictor",
    page_icon="💻",
    layout="centered"
)

# CSS for desktop-centered layout and simple theme
st.markdown("""
    <style>
        .block-container {
            padding-top: 6rem;
            padding-bottom: 6rem;
            max-width: 700px;
            margin: auto;
        }
        h1.title {
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
            font-size: 2.8rem;
            color: #1e2b5c;
            margin-bottom: 0.3rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2.5rem;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #999;
            margin-top: 0rem;
            margin-bottom: 4rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<h1 class="title">Face-to-BMI Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a face image to estimate BMI using deep learning</div>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📸 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    st.markdown("🎯 **Predicting BMI...**")
    with st.spinner("Analyzing facial features..."):
        # Dummy model simulation
        dummy_bmi = round(np.random.normal(25, 4), 1)

        if dummy_bmi < 18.5:
            category = "Underweight"
            emoji = "🔵"
        elif dummy_bmi < 25:
            category = "Normal"
            emoji = "🟢"
        elif dummy_bmi < 30:
            category = "Overweight"
            emoji = "🟡"
        else:
            category = "Obese"
            emoji = "🔴"

        st.success(f"✅ Predicted BMI: **{dummy_bmi}**  —  {emoji} **{category}**")

# Footer
st.markdown(
    """
    <div class="footer">
        <hr style="margin-top: 3rem; margin-bottom: 1.5rem; border: none; height: 1px; background-color: #ccc;" />
        🧠 Created for the Machine Learning 2 Project<br>
        © 2025 MSADS at The University of Chicago
    </div>
    """,
    unsafe_allow_html=True
)
