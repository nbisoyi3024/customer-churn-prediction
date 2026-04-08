from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from src.utils import load_model

import signal
import sys

#Load your saved model using pickle.load()
with open("models/Random Forest.pkl", "rb") as file:
    model = pickle.load(file)
print(type(model))

#create a flask app
app = Flask(__name__)


@app.route('/')
def home():
    """Simple home page"""
    return """
    <h1>Customer Churn Prediction API</h1>
    <p>Welcome to the Customer Churn Prediction API!</p>

    <h2>How to use:</h2>
    <p>Send a POST request to <code>/predict</code> with customer features:</p>

    <h3>Example:</h3>
    <pre>
    {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 35,
        "Tenure": 3,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 60000,
        "Card Type": "Gold"
    }
    </pre>
    
      <h3>Features:</h3>
    <ul>
        <li><strong>CreditScore</strong>: Average  Credit Score</li>
        <li><strong>Geography</strong>:Geography(France/Spain/Germany)</li>
        <li><strong>Gender</strong>: Gender (M/F)</li>
        <li><strong>Age</strong>: Age(15-100) (0-10)</li>
        <li><strong>Balance</strong>: Balance(2000-200000)</li>
        <li><strong>No of Products</strong>: No Of Products(1-5)</li>
        <li><strong>Credit Card</strong>: Credit Card(0/1)</li>
        <li><strong>Active Member</strong>: Active Member(0/1)</li>
        <li><strong>Estimated Salary</strong>: Estimated Salary(2000-200000)</li>
        <li><strong>Card Type</strong>: Card Type</li>
    </ul>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Make churn prediction"""
    try:
        data = request.get_json()

        # Convert Json to DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return jsonify({
            'churn_prediction': int(prediction),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Closing the Flask app
def signal_handler(sig, frame):
    print('\nShutting down Flask app...')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("Starting Customer Churn Flask API...")
    print("Visit http://127.0.0.1:5000 for usage instructions")
    app.run(debug=True, port=5000, use_reloader=False)