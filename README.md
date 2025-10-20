📩 SMS Spam Detector (Unique Edition)A robust and high-precision Machine Learning application designed to classify text messages as Spam or Ham (Legitimate). This project stands out by integrating advanced feature engineering and classifier tuning to excel at detecting regional/slang-based spam.

🌐 Live Demo & Deployment
The application successfully runs in a containerized cloud environment, bypassing local port issues.

👉 [https://sms-spam-detector-7elcpo6djsrmlkrvbdmh5b.streamlit.app/]

🌟 Project HighlightsCustom Hybrid Model: Deployed a Logistic Regression pipeline, which is superior for this classification task compared to a simple Naive Bayes baseline1.Unique Data Augmentation: The model was trained on a large public dataset merged with custom-curated regional data to accurately identify Indian financial spam keywords (e.g., "Lakhs," "Rupees," "KYC")2.High Precision Tuning (The Fix): The model was specifically tuned using the  filter and Handcrafted Features to eliminate False Positives on tricky, conversational message.Explainable UI: The Streamlit demo provides a visualization of the decision by highlighting spam-related words present in the input text.

Component,Tools Used

Language,Python 
Machine Learning,  ,"scikit-learn (Logistic Regression, ColumnTransformer) "
Feature Extraction,,  "TfidfVectorizer, Handcrafted Features "
Web / UI,  ,Streamlit (for the easy web demo) 
Model Persistence,  ,joblib 



