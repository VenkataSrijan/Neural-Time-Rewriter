# backend/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Columns that are plain numbers — we scale these
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Columns that are categories — we leave these as-is
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# All 13 input columns (everything except target)
FEATURE_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol',
                'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal']


def load_data(filepath='data/heart.csv'):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    X = df[FEATURE_COLS].copy()
    y = df['target'].copy()
    return X, y, df


def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X[NUMERICAL_COLS])
    os.makedirs('backend/models', exist_ok=True)
    joblib.dump(scaler, 'backend/models/scaler.pkl')
    print("Scaler saved!")
    return scaler


def transform(X, scaler):
    X_copy = X.copy()
    X_copy[NUMERICAL_COLS] = scaler.transform(X_copy[NUMERICAL_COLS])
    return X_copy.values.astype(np.float32)


def inverse_transform_row(row_array, scaler):
    row = pd.DataFrame([row_array], columns=FEATURE_COLS)
    row[NUMERICAL_COLS] = scaler.inverse_transform(row[NUMERICAL_COLS])
    return row.iloc[0].to_dict()


def get_feature_ranges(X):
    ranges = {}
    for col in FEATURE_COLS:
        ranges[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'type': 'numerical' if col in NUMERICAL_COLS else 'categorical'
        }
    return ranges