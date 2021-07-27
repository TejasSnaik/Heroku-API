import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from scipy import stats

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(i) for i in request.form.values()]

    final_features = [np.array(features)]

    output = stats.mode(model.predict(final_features))
    if output[0][0] == 1:
        return render_template("index.html", predicted_text="Yes Diagnose of heart disease is needed")
    else:
        return render_template("index.html", predicted_text="No need of diagnose for Heart Disease")


if __name__ == "__main__":
    app.run(debug=True)