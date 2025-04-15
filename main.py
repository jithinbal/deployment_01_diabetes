import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scalar = joblib.load("scalar.pkl")


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/submit", methods=["post"])
def submit():
    data = [eval(data) for data in request.form.values()]
    data_array = np.array([data])
    scaled_input = scalar.transform(data_array)
    result = model.predict(scaled_input)
    if result[0] == 1:
        return "diabetic"
    else:
        return "not diabetic"


app.run(host="0.0.0.0", port=7080, debug=True)
