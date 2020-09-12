from flask import Flask, jsonify, request
import pandas as pd
app = Flask(__name__)

@app.route('/home', methods=['GET'])
def get_data():
    some_data = pd.read_csv('../ehr_diabetes_no_missing_3k.csv')
    response = {'diabetes_data':some_data.head().to_json()}
    return jsonify(response)