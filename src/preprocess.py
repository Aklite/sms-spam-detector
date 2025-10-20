# src/preprocess.py
import re
import pandas as pd
import os

# --- Cleaning Function ---
abbr = {' u ':' you ', ' ur ':' your ', ' r ':' are ', ' pls ':' please ', ' msg ':' message '}

def clean_text(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', ' URL ', s)
    s = re.sub(r'\d+', ' NUM ', s)
    for k,v in abbr.items():
        s = s.replace(k, v)
    s = re.sub(r'[^a-zA-Z0-9\s!?.]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --- File Handling (Using Absolute Paths) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw_sms.csv')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed_sms.csv')

try:
    df = pd.read_csv(RAW_PATH)
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    df.to_csv(PROCESSED_PATH, index=False)
    print("✅ Preprocessing done. Saved to data/processed_sms.csv")
except FileNotFoundError:
    print(f"🛑 CRITICAL ERROR: Raw data file not found at {RAW_PATH}. Ensure raw_sms.csv exists.")
except Exception as e:
    print(f"🛑 CRITICAL ERROR: Preprocessing failed due to: {e}")