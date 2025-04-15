import joblib
import numpy as np

scalar = joblib.load("scalar.pkl")
model = joblib.load("model.pkl")
user_input = np.array([[6, 148, 72, 35, 94, 33.6, 0.627, 50]])
scaled_input = scalar.transform(user_input)
result = model.predict(scaled_input)
print(f"result is = {result[0]}")
print(result)  # [1] or [0]
