# app/app.py (Final Polished Version)
import streamlit as st
import joblib
import pandas as pd
import re
import os
from io import StringIO

# --- File Path Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pipeline.joblib')


# --- Feature Calculation (Must match the training pipeline) ---
def clean_text(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www.\S+', ' URL ', s)
    s = re.sub(r'\d+', ' NUM ', s)
    s = s.replace(' u ', ' you ').replace(' pls ', ' please ').replace(' r ', ' are ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def calculate_features(text):
    """Calculates all features (raw and clean) needed for the pipeline."""
    text_str = str(text)
    
    # Calculate Handcrafted Features (Must be correct for StandardScaler input)
    msg_len = len(text_str)
    num_exclaim = text_str.count('!')
    has_url = 1 if 'http' in text_str or 'www' in text_str else 0
    upper_ratio = sum(1 for c in text_str if c.isupper()) / max(1, len(text_str))
    
    data = {
        'clean_text': [clean_text(text)],
        'msg_len': [msg_len],
        'num_exclaim': [num_exclaim],
        'has_url': [has_url],
        'upper_ratio': [upper_ratio]
    }
    return pd.DataFrame(data)

# --- Load Model ---
try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Model pipeline.joblib not found. Please run notebooks/02_train.py first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- Streamlit UI Setup ---
st.set_page_config(page_title="SMS Spam Detector", page_icon=" 📱 ", layout="centered")
st.markdown("""
<h1 style='text-align: center; color: #1a73e8;'> 📩 SMS Spam Detector (Unique Edition)</h1>
<p style='text-align: center;'>Logistic Regression Model combining TF-IDF and Handcrafted Features.</p>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("Project Details")
st.sidebar.info("""
**Developer:** AK
**Model:** Logistic Regression (Tuned for Precision)
**Features:** TF-IDF + Custom (Length, Exclaims, URL, Upper Ratio)
""")

# Text input area
message = st.text_area(" ✉️ Enter your message below:", height=120)

if st.button(" 🔍 Predict"):
    if message.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        # Prepare data for the full pipeline
        input_data = calculate_features(message)
        
        # Predict using the full pipeline
        result = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0]
        
        # Get confidence for the predicted class
        pred_index = list(pipeline.classes_).index(result)
        confidence = prob[pred_index]

        # Color result
        color = "red" if result == "spam" else "green"
        st.markdown(f"<h3 style='color:{color};'>Prediction: {result.upper()}</h3>", unsafe_allow_html=True)
        st.progress(confidence)
        st.caption(f"Confidence: {confidence:.2f}")

        # Highlight spammy words (Unique Bonus Feature)
        spam_words = ["win", "free", "claim", "offer", "cash", "urgent", "click", "prize", "reward", "money", "rs", "rupees", "lakh", "lakhs", "bonus", "income", "profit", "call", "account", "kyc", "bank"]
        found = [w for w in message.lower().split() if w in spam_words]
        
        if found:
            st.info("Spam-related words detected: " + ", ".join(found))
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("Developed by ARUNKUMAR | Powered by Streamlit & scikit-learn")