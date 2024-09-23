from flask import Blueprint, jsonify
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
import  googlemaps
import pandas as pd
import csv
from dotenv import load_dotenv
import os

# Define a blueprint
main = Blueprint('main', __name__)


# model_path = os.path.join(os.path.dirname(__file__), '../model/rf.pkl')
# model = joblib.load(model_path)

# model_path = os.path.join(os.path.dirname(__file__), '../model/rf.pkl')
dosha_model_path = os.path.join(os.path.dirname(__file__), '../model/Dosha/Dosha_Prediction_Model.pkl')
risk_model_path = os.path.join(os.path.dirname(__file__), '../model/Risk/Risk_Prediction_Model.pkl')
medicine_model_path = os.path.join(os.path.dirname(__file__), '../model/Medicine/Medicine_Prediction_Model.pkl')

# Load models
# model = joblib.load(model_path)
dosha_model = joblib.load(dosha_model_path)
risk_model = joblib.load(risk_model_path)
medicine_model = joblib.load(medicine_model_path)

# Construct encoder paths
dosha_encoder_path = os.path.join(os.path.dirname(__file__), '../model/Dosha/Dosha_LabelEncoder.pkl')
risk_encoder_path = os.path.join(os.path.dirname(__file__), '../model/Risk/Risk_LabelEncoder.pkl')
medicine_encoder_path = os.path.join(os.path.dirname(__file__), '../model/Medicine/Medicine_LabelEncoder.pkl')

# Load encoders if needed
dosha_encoder = joblib.load(dosha_encoder_path)
risk_encoder = joblib.load(risk_encoder_path)
medicine_encoder = joblib.load(medicine_encoder_path)



# model = joblib.load('model/rf.pkl')
# dosha_model = joblib.load('model/Dosha/Dosha_Prediction_Model.pkl')
# risk_model = joblib.load('model/Risk/Risk_Prediction_Model.pkl')
# medicine_model = joblib.load('model/Medicine/Medicine_Prediction_Model.pkl')

# # Load encoders if needed 
# dosha_encoder = joblib.load('model/Dosha/Dosha_LabelEncoder.pkl')
# risk_encoder = joblib.load('model/Risk/Risk_LabelEncoder.pkl')
# medicine_encoder = joblib.load('model/Medicine/Medicine_LabelEncoder.pkl')

@main.route('/')
def home():
    return "Welcome to the Disease Prediction API!"

@main.route('/symptoms', methods=['GET'])
def symptomDetails():
    file_path = 'db/symptomsnew.json'
    with open(file_path, 'r') as symptom_json:
        symptoms = json.load(symptom_json)
    return jsonify(symptoms)

@main.route('/symptomsearch', methods=['POST'])
def symptomSearch():
    request_data = request.get_json()

    if 'search_key' in request_data:
        search_key = request_data['search_key'].lower()

        file_path = 'db/symptoms.json'
        with open(file_path, 'r') as symptom_json:
            symptoms = json.load(symptom_json)

        filtered_symptoms = [symptom for symptom in symptoms if search_key in symptom['symptom'].lower() or search_key in symptom['description'].lower()]
        return jsonify(filtered_symptoms), 200
    
    return jsonify({'message': 'Please provide a search_key in the request body.'}), 400

@main.route('/feedback', methods=['POST'])
def saveFeedback():
    request_data = request.get_json()
    
    # Create a dictionary to hold the feedback data
    feedback_data = {
        'name': request_data.get('name'),
        'email': request_data.get('email'),
        'message': request_data.get('message'),
        'rating': request_data.get('rating')
    }

    # Save the feedback data to a CSV file
    with open('db/feedback.csv', 'a', newline='') as file:
        fieldnames = ['name', 'email', 'message', 'rating']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(feedback_data)

    return jsonify({'message': 'Feedback saved successfully.'}), 200

# @main.route('/predict', methods=['POST'])
# def getPredictions():
#     num_features = model.n_features_in_
#     custom_array = np.zeros(num_features)
#     symptom_ids = [int(x) for x in request.json['ids']]

#     for id in symptom_ids:
#         custom_array[id] = 1
    
#     prob = model.predict_proba([custom_array])[0]
#     prediction_classes = model.classes_
    
#     threshold = 0.01
#     predictions_with_prob = [{'disease': label, 'probability': float(probability)} for label, probability in zip(prediction_classes, prob) if probability > threshold]
    
#     predictions_with_prob = sorted(
#         predictions_with_prob,
#         key=lambda x: x['probability'],
#         reverse=True
#     )

#     response = predictions_with_prob
    
#     return jsonify(response), 200


# Route for Dosha Prediction
@main.route('/predict/dosha', methods=['POST'])
def predictDosha():
    num_features = dosha_model.n_features_in_
    custom_array = np.zeros(num_features)
    symptom_ids = [int(x) for x in request.json['ids']]

    for id in symptom_ids:
        custom_array[id] = 1

    # Make prediction and decode result
    dosha_prediction = dosha_model.predict([custom_array])[0]
    dosha_label = dosha_encoder.inverse_transform([dosha_prediction])[0]

    return jsonify({'dosha': dosha_label}), 200

# Route for Risk Prediction
@main.route('/predict/risk', methods=['POST'])
def predictRisk():
    num_features = risk_model.n_features_in_
    custom_array = np.zeros(num_features)
    symptom_ids = [int(x) for x in request.json['ids']]

    for id in symptom_ids:
        custom_array[id] = 1

    # Make prediction and decode result
    risk_prediction = risk_model.predict([custom_array])[0]
    risk_label = risk_encoder.inverse_transform([risk_prediction])[0]

    return jsonify({'risk': risk_label}), 200

# Route for Medicine Prediction
@main.route('/predict/medicine', methods=['POST'])
def predictMedicine():
    num_features = medicine_model.n_features_in_
    custom_array = np.zeros(num_features)
    symptom_ids = [int(x) for x in request.json['ids']]

    for id in symptom_ids:
        custom_array[id] = 1

    # Make prediction and decode result
    medicine_prediction = medicine_model.predict([custom_array])[0]
    medicine_label = medicine_encoder.inverse_transform([medicine_prediction])[0]

    return jsonify({'medicine': medicine_label}), 200
