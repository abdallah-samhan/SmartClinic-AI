import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the dataset
try: 
    data_set = pd.read_csv('C:\\xampp\\htdocs\\web_doctor-main\\web_doctor-diabetes\\diabetes_prediction\\diabetes.csv')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()    

# 2. Encode categorical column
data_set['gender'] = data_set['gender'].map({'Male': 0, 'Female': 1})

# 3. Select relevant columns
cols = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'blood_glucose_level']

# 4. Prepare features and labels
X = data_set[cols].values
y = data_set.iloc[:, -1].values  # last column is the target

# 5. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42
)

# 6. Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train the model with class_weight balanced
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',  # <--- This line handles class imbalance
    random_state=42
)
clf.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = clf.predict(X_test)
print("\n" + "="*50)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("="*50 + "\n")

# 9. Save the model and scaler
joblib.dump(clf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully!")

# 10. Test the model on new data
test_samples = [
    [0, 44.0, 0, 0, 19.31, 200],  # Female → 0
    [0, 80.0, 0, 1, 25.19, 140]
]
print("Test results:")
for sample in test_samples:
    sample_scaled = scaler.transform([sample])
    prediction = clf.predict(sample_scaled)[0]
    print(f"Sample: {sample[:3]}... → Prediction: {'Diabetic' if prediction == 1 else 'Healthy'}")
