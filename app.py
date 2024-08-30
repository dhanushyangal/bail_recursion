from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load the model, scaler, and label encoders
model = joblib.load('bail_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load dataset for lookup
data = pd.read_csv('bail_decision_data.csv')

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
    return jsonify(defendant)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    defendant = data[data['Name'] == name]
    
    if defendant.empty:
        return jsonify({'error': 'Defendant not found'}), 404

    defendant = defendant.iloc[0].drop(labels=['Name', 'Bail_Granted']).to_dict()

    # Encode and scale features
    for column in label_encoders:
        defendant[column] = label_encoders[column].transform([defendant[column]])[0]
    
    features = scaler.transform([list(defendant.values())])

    # Predict bail eligibility
    prediction = model.predict(features)[0]
    
    return jsonify({'bail_eligibility': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
