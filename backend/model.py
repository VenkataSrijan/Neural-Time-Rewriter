# backend/model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import numpy as np


def train_classifier(X_scaled, y):
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Random Forest trains in seconds and is very accurate
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Check how good it is
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save the model
    joblib.dump(clf, 'backend/models/classifier.pkl')
    print("Classifier saved!")

    return clf


def load_classifier():
    return joblib.load('backend/models/classifier.pkl')


def predict(clf, X_scaled_row):
    # X_scaled_row is a single row as numpy array
    row = X_scaled_row.reshape(1, -1)
    prediction = clf.predict(row)[0]
    probability = clf.predict_proba(row)[0]
    return {
        'prediction': int(prediction),
        'label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'confidence': round(float(max(probability)) * 100, 2)
    }