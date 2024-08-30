import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('bail_decision_data.csv')

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Offense_Type', 'Community_Ties', 'Employment_Status', 'Criminal_History_Severity']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split features and labels
X = data.drop(columns=['Name', 'Bail_Granted'])  # Features (excluding 'Name' and 'Bail_Granted')
y = data['Bail_Granted']  # Label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model development (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and encoders
joblib.dump(model, 'bail_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
