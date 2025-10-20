# src/features.py
import pandas as pd
import os

# --- File Handling (Using Absolute Paths) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed_sms.csv')

try:
    df = pd.read_csv(PROCESSED_PATH)
    
    # --- Handcrafted Features ---
    df['msg_len'] = df['text'].astype(str).str.len()
    df['num_exclaim'] = df['text'].str.count('!')
    df['has_url'] = df['text'].str.contains('http|www|URL').fillna(False).astype(int)
    df['upper_ratio'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(1,len(str(x))))
    
    df.to_csv(PROCESSED_PATH, index=False)
    print("✅ Features added and saved again.")
except FileNotFoundError:
    print(f"🛑 CRITICAL ERROR: Processed data file not found at {PROCESSED_PATH}. Run preprocess.py first.")
except Exception as e:
    print(f"🛑 CRITICAL ERROR: Feature calculation failed due to: {e}")