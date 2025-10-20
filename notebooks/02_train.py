# notebooks/02_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os 

# --- File Handling (Using Absolute Paths) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed_sms.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pipeline.joblib')

try:
    df = pd.read_csv(PROCESSED_PATH)
except FileNotFoundError:
    print("🛑 CRITICAL ERROR: Data file processed_sms.csv not found. Run preprocess.py and features.py first.")
    exit()

# --- 1. Define Features and Target ---
NUMERIC_FEATURES = ['msg_len', 'num_exclaim', 'has_url', 'upper_ratio']
TEXT_FEATURE = 'clean_text'
TARGET = 'label'

X = df[[TEXT_FEATURE] + NUMERIC_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 2. Build the Preprocessing Transformer (Merges two paths) ---
preprocessor = ColumnTransformer(
    transformers=[
        # CRITICAL FIX: max_df=0.7 aggressively filters highly common Ham words 
        ('text_pipe', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.7), TEXT_FEATURE),
        # Path 2: Scale the numeric/handcrafted features
        ('num_pipe', StandardScaler(), NUMERIC_FEATURES)
    ],
    remainder='drop'
)

# --- 3. Define the Final Pipeline (Preprocessing + Model) ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Use Logistic Regression with regularization (C=10) for better precision
    ('classifier', LogisticRegression(C=10, solver='liblinear', random_state=42))
])

# --- 4. Train the Model ---
pipeline.fit(X_train, y_train)

# --- 5. Evaluate and Save ---
pred = pipeline.predict(X_test)
print("-------------------------------------------------------")
print("CLASSIFICATION REPORT (Final Logistic Regression Model):")
print(classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Final Combined Pipeline saved to {MODEL_PATH}")
print("-------------------------------------------------------")