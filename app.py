from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import pandas as pd
import joblib

# Load model, scaler, and encoders
model = joblib.load('bail_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load dataset for lookup
data = pd.read_csv('bail_decision_data.csv')

# List of feature columns used for prediction
feature_columns = ['Gender', 'Offense_Type', 'Community_Ties', 'Employment_Status', 'Criminal_History_Severity', 'Release_Section']

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required to store session data

# Predefined users with roles and passwords
users = {
    'lawyer': {'username': 'lawyer123', 'password': 'lawyerpass'},
    'judge': {'username': 'judge123', 'password': 'judgepass'}
}

# A list to store submitted bail requests (for simplicity, using an in-memory list)
bail_requests = []

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']

    # Authenticate user based on role
    if role in users and username == users[role]['username'] and password == users[role]['password']:
        session['role'] = role
        if role == 'lawyer':
            return redirect(url_for('lawyer_dashboard'))
        elif role == 'judge':
            return redirect(url_for('judge_dashboard'))
    else:
        return "Invalid credentials. Please try again."

@app.route('/lawyer')
def lawyer_dashboard():
    if 'role' in session and session['role'] == 'lawyer':
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/judge')
def judge_dashboard():
    if 'role' in session and session['role'] == 'judge':
        return render_template('judge_dashboard.html', requests=bail_requests)
    else:
        return redirect(url_for('login'))

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
            features.append(0)  # Default value if data is missing
    
    features = [features]
    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    
    # Build result
    reasons = {}
    if prediction == 1:
        reasons['bail_eligibility'] = 'Bail Granted'
        reasons['reason'] = 'The defendant meets the criteria for bail based on legal and procedural parameters.'
    else:
        reasons['bail_eligibility'] = 'Bail Denied'
        reasons['reason'] = 'The defendant does not meet the criteria for bail. Possible reasons: flight risk, serious offense, etc.'

    result = {
        'bail_eligibility': reasons['bail_eligibility'],
        'reason': reasons['reason'],
        'release_section': defendant.get('Release_Section', 'N/A'),
        'reason_for_bail_denial': defendant.get('Reason_for_Bail_Denial', 'N/A') if prediction == 0 else 'N/A'
    }

    return render_template('result.html', result=result)

@app.route('/submit_to_judge', methods=['POST'])
def submit_to_judge():
    name = request.form['name']
    bail_status = request.form['bail_eligibility']
    release_section = request.form['release_section']

    # Add to bail requests for judge's review
    bail_requests.append({
        'name': name,
        'bail_eligibility': bail_status,
        'release_section': release_section,
        'status': 'Pending'
    })
    
    return redirect(url_for('login'))


@app.route('/judge_review', methods=['POST'])
def judge_review():
    index = int(request.form['index'])
    action = request.form['action']
    
    # Update the request based on judge's decision
    if action == 'approve':
        bail_requests[index]['status'] = 'Approved'
    elif action == 'reject':
        bail_requests[index]['status'] = 'Rejected'
    
    return redirect(url_for('judge_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
