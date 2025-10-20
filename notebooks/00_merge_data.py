import pandas as pd
from io import StringIO
import os

# Define the absolute path to the data file: D:\Sms\data\public_sms.csv
DATA_PATH = os.path.join(os.getcwd(), 'data', 'public_sms.csv')

# --- Step 1: Load the Large Public Dataset (5574+ rows) ---
try:
    # Try reading with common column names and encoding for the UCI dataset
    public_df = pd.read_csv(DATA_PATH, encoding='latin-1', sep='\t', header=None, names=['label', 'text'])
except:
    # Fallback to simple CSV read (if the file was already converted to CSV)
    public_df = pd.read_csv(DATA_PATH, encoding='latin-1', header=None, names=['label', 'text'])

# ... rest of the script is the same ...