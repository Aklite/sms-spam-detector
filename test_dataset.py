import pandas as pd
df = pd.read_csv('data/raw_sms.csv')
print(df.head())
print(df['label'].value_counts())
