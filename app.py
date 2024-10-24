from flask import Flask, request, jsonify
import pickle
import io
import os
import pandas as pd
import json
from diabetes_detection_nn import load_dataset, get_train_test_data, train_NN_model, feature_scaling
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)
BLOB_CONTAINER_NAME = os.getenv('BLOB_CONTAINER_NAME')
BLOB_NAME = os.getenv('MODEL_FILENAME')
CONNECTION_STRING = os.getenv('BLOB_CONNECTION_STRING')


@app.route('/predict', methods=['POST'])
def predict():
    # Load model
    saved_model = load_model_from_blob()
    data = request.json
    print("User provided inputs:", data)
    processed_input_df = format_user_input(data)
    prediction = saved_model.predict(processed_input_df)
    return jsonify({"prediction": prediction_stringify(int(prediction[0]))})


@app.route('/train', methods=['POST'])
def train():
    NN_model = train_NN_model()
    # Save the trained model to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump(NN_model, f)
    # saves model as a file in Azure Blob Storage
    return upload_model_to_storage()


@app.route('/assert', methods=['GET'])
def assert_stratification():
    X_train, X_test, y_train, y_test, y = get_train_test_data()
    original_proportion = y.value_counts(normalize=True)
    train_proportion = y_train.value_counts(normalize=True)
    test_proportion = y_test.value_counts(normalize=True)

    # Assert that the difference in proportions is small
    assert (original_proportion - train_proportion).abs().max() < 0.05, "Training set stratification failed"
    assert (original_proportion - test_proportion).abs().max() < 0.05, "Testing set stratification failed"

    message = "Assertion passed: The train and test sets are properly stratified."
    if not is_stratified(original_proportion, train_proportion, test_proportion):
        message = "Proportion difference exceeds the threshold."

    proportion_data = {
        "message":  message
    }
    print(proportion_data)
    # Convert the dictionary to a JSON string
    json_result = json.dumps(proportion_data, indent=4)

    return json_result

def is_stratified(original_proportion, train_proportion, test_proportion):
    if not (original_proportion - train_proportion).abs().max() < 0.05:
        return False
    if not (original_proportion - test_proportion).abs().max() < 0.05:
        return False
    return True

def prediction_stringify(prediction_value):
    if prediction_value == 0:
        return "No Diabetes found"
    else:
        return "Diabetes is found"

def load_model_from_blob():
    try:
        # Connect to the Blob service
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        
        # Get a client for the container and blob
        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=BLOB_NAME)
        
        # Download the blob (model) as a stream
        download_stream = blob_client.download_blob()
        model_bytes = download_stream.readall()
        
        # Load the model from the byte stream
        model = pickle.load(io.BytesIO(model_bytes))
        
        return model
    except Exception as e:
        print(f"Error loading model from blob: {e}")
        try:
            print("Fall back to loading the local model.pkl since blob download failed")
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
                print("Loaded model from local 'model.pkl'")
                return model
        except Exception as local_e:
            print(f"Error loading model from local file: {local_e}")
            return None

def upload_model_to_storage():
    try:
        # Initialize a connection to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        
        # Get a client to interact with the container
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        
        # Create the container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
        
        # Get a client to interact with the blob (model file)
        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=BLOB_NAME)
        
        # Open the local model file and upload it to the blob
        with open('model.pkl', 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"Model uploaded successfully to container '{BLOB_CONTAINER_NAME}' as blob '{BLOB_NAME}'")
        return "Success"
    except Exception as e:
        print(f"Failed to upload model: {str(e)}")
        return "Failed" 

# formats user input in the format - `{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}` 
# to dataframe and then scales it as necessary to be fit to be sent to the predict endpoint
def format_user_input(input_data):
    df = pd.DataFrame([input_data])
    # Columns with comma decimal
    columns_with_comma_decimal = ['chol_hdl_ratio', 'bmi', 'waist_hip_ratio']
    for column in columns_with_comma_decimal:
        df[column] = df[column].astype(str).str.replace(',', '.').astype(float)

    # Encode 'gender' column
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})

    # Handle missing values (if any)
    df.fillna(df.median(), inplace=True)

    # Apply feature scaling
    scaled_data, _ = feature_scaling(df)

    return scaled_data

@app.route('/data', methods=['GET'])
def data_head():
    return load_dataset().head().to_json(orient='records')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)