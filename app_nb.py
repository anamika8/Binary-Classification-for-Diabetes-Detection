from flask import Flask, request, jsonify
import pickle
import io
import os
import pandas as pd
import json
from azure.storage.blob import BlobServiceClient
from common import load_dataset

app = Flask(__name__)

@app.route('/data-nb', methods=['GET'])
def data_head():
    return load_dataset().head().to_json(orient='records')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)