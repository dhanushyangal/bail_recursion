import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Load the model, scaler, and label encoders
model = joblib.load('bail_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load dataset for lookup
data = pd.read_csv('bail_decision_data.csv')

# List of feature columns used for training the model
feature_columns = ['Gender', 'Offense_Type', 'Community_Ties', 'Employment_Status', 'Criminal_History_Severity', 'Release_Section']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_details', methods=['POST'])
def get_details():
    name = request.form['name']
    defendant = data[data['Name'] == name]
    
    if defendant.empty:
        return jsonify({'error': 'Defendant not found'}), 404

    defendant = defendant.iloc[0].to_dict()
    return render_template('details.html', details=defendant)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    defendant = data[data['Name'] == name]
    
    if defendant.empty:
        return jsonify({'error': 'Defendant not found'}), 404

    defendant = defendant.iloc[0]
    
    # Prepare features for prediction
    features = []

    for column in feature_columns:
        if column in defendant:
            value = defendant[column]
            if column in label_encoders:
                value = label_encoders[column].transform([value])[0]
            features.append(value)
        else:
            # Append default value for missing features
            features.append(0)  # or another suitable default value
    
    features = [features]  # Make it a 2D array for scaler and model

    # Scale and predict
    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    
    # Get reasons for bail decision
    reasons = {}
    if prediction == 1:
        reasons['bail_eligibility'] = 'Bail Granted'
        reasons['reason'] = 'The defendant meets the criteria for bail based on the current legal and procedural parameters.'
    else:
        reasons['bail_eligibility'] = 'Bail Denied'
        reasons['reason'] = 'The defendant does not meet the criteria for bail. Reasons may include high flight risk, serious nature of the offense, or other factors.'

    # Add detailed reasons
    result = {
        'bail_eligibility': reasons['bail_eligibility'],
        'reason': reasons['reason'],
        'release_section': defendant.get('Release_Section', 'N/A'),
        'reason_for_bail_denial': defendant.get('Reason_for_Bail_Denial', 'N/A') if prediction == 0 else 'N/A'
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
