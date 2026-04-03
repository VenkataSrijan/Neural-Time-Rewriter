# backend/app.py

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

from backend.preprocess import (
    load_data, transform, get_feature_ranges,
    FEATURE_COLS, NUMERICAL_COLS
)
from backend.model import load_classifier, predict
from backend.vae import load_vae
from backend.counterfactual import generate_counterfactual

app = Flask(__name__)
CORS(app)  # Allows frontend to talk to backend

# ── Load everything once when server starts ──────────────────────────
print("Loading models...")
scaler     = joblib.load('backend/models/scaler.pkl')
classifier = load_classifier()
vae        = load_vae()

X, y, df   = load_data()
X_scaled   = transform(X, scaler)
ranges     = get_feature_ranges(X)
print("All models loaded! Server ready.")


# ── Route 1: Health check ─────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({'status': 'Neural Time Rewriter API is running!'})


# ── Route 2: Get feature ranges (for frontend sliders) ───────────────
@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify(ranges)


# ── Route 3: Get a sample patient from dataset ───────────────────────
@app.route('/api/sample', methods=['GET'])
def get_sample():
    index = np.random.randint(0, len(X))
    row   = X.iloc[index].to_dict()
    row   = {k: round(float(v), 2) for k, v in row.items()}
    return jsonify({'index': int(index), 'data': row})


# ── Route 4: Predict outcome for given patient data ──────────────────
@app.route('/api/predict', methods=['POST'])
def predict_route():
    body = request.get_json()
    data = body.get('data')           # dict of feature: value

    # Convert to DataFrame then scale
    import pandas as pd
    row_df     = pd.DataFrame([data])[FEATURE_COLS]
    row_scaled = transform(row_df, scaler)[0]

    result = predict(classifier, row_scaled)
    return jsonify(result)


# ── Route 5: Generate counterfactual ─────────────────────────────────
@app.route('/api/counterfactual', methods=['POST'])
def counterfactual_route():
    body            = request.get_json()
    data            = body.get('data')            # original patient data (raw)
    changed_feature = body.get('changed_feature') # e.g. "trestbps"
    new_value       = float(body.get('new_value'))# e.g. 120

    # Scale the original row
    import pandas as pd
    row_df     = pd.DataFrame([data])[FEATURE_COLS]
    row_scaled = transform(row_df, scaler)[0]

    result = generate_counterfactual(
        original_row    = row_scaled,
        changed_feature = changed_feature,
        new_value       = new_value,
        scaler          = scaler,
        classifier      = classifier,
        vae             = vae
    )
    return jsonify(result)


# ── Run the server ────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)