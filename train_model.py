import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('bail_decision_data.csv')

# Handle missing values
data.fillna({
    'Release_Section': 'Unknown',
    'Reason_for_Bail_Denial': 'Not Applicable',
    'Penal_Provisions': 'Unknown',
    'Duration_of_Imprisonment_Served': 'Unknown',
    'Risk_of_Tampering_with_Evidence_or_Witnesses': 'Unknown',
    'Socio-economic_Background': 'Unknown',
    'Health_and_Personal_Circumstances': 'Unknown',
    'Compliance_with_Procedural_Aspects': 'Unknown',
    'Judicial_Pronouncements': 'Unknown'
}, inplace=True)

# Encode categorical variables
label_encoders = {}
feature_columns = [
    'Gender', 'Offense_Type', 'Community_Ties', 'Employment_Status',
    'Criminal_History_Severity', 'Release_Section', 'Penal_Provisions',
    'Risk_of_Tampering_with_Evidence_or_Witnesses', 'Socio-economic_Background',
    'Health_and_Personal_Circumstances', 'Compliance_with_Procedural_Aspects',
    'Judicial_Pronouncements'
]

for column in feature_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Split features and labels
X = data[feature_columns]
y = data['Bail_Granted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model development
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and encoders
joblib.dump(model, 'bail_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
