from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
from diabetes_detection_nn import load_dataset, get_train_test_data

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

@app.route('/assert', methods=['GET'])
def assert_stratification():
    X_train, X_test, y_train, y_test, y = get_train_test_data()
    original_proportion = y.value_counts(normalize=True)
    train_proportion = y_train.value_counts(normalize=True)
    test_proportion = y_test.value_counts(normalize=True)

    # Assert that the difference in proportions is small
    assert (original_proportion - train_proportion).abs().max() < 0.05, "Training set stratification failed"
    assert (original_proportion - test_proportion).abs().max() < 0.05, "Testing set stratification failed"

    proportion_data = {
    "original_proportion": original_proportion,
    "train_proportion": train_proportion,
    "test_proportion": test_proportion,
    "message": "Assertion passed: The train and test sets are properly stratified." if ((abs(original_proportion - train_proportion) < 0.05) and (abs(original_proportion - train_proportion) < 0.05)) else "Proportion difference exceeds the threshold."
    }

    # Convert the dictionary to a JSON string
    json_result = json.dumps(proportion_data, indent=4)

    return json


@app.route('/data', methods=['GET'])
def data_head():
    return load_dataset().head().to_json(orient='records')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)