from flask import Flask, request, jsonify
import pickle
import pandas as pd
from diabetes_detection_nn import load_dataset

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"prediction": int(prediction[0])})

@app.route('/data-head', methods=['GET'])
def predict():
    return load_dataset()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)