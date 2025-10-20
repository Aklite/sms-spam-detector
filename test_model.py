import joblib

model = joblib.load('models/mnb_pipeline.joblib')

while True:
    msg = input("Enter a message (or 'quit' to exit): ")
    if msg.lower() == 'quit':
        break
    pred = model.predict([msg])[0]
    print(f"Prediction: {pred}")

