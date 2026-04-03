# backend/counterfactual.py

import numpy as np
import pandas as pd
from backend.preprocess import FEATURE_COLS, NUMERICAL_COLS, inverse_transform_row


def generate_counterfactual(original_row, changed_feature, new_value,
                             scaler, classifier, vae):

    # Step 1: Readable original
    original_readable = inverse_transform_row(original_row.copy(), scaler)

    # Step 2: Change only the selected feature
    cf_row = original_row.copy()
    feature_index = FEATURE_COLS.index(changed_feature)

    if changed_feature in NUMERICAL_COLS:
        temp = pd.DataFrame([original_readable])
        temp[changed_feature] = new_value
        scaled_numerical = scaler.transform(temp[NUMERICAL_COLS])[0]
        num_index = NUMERICAL_COLS.index(changed_feature)
        cf_row[feature_index] = scaled_numerical[num_index]
    else:
        cf_row[feature_index] = float(new_value)

    # Step 3: Predict
    orig_pred = classifier.predict(original_row.reshape(1, -1))[0]
    orig_prob = classifier.predict_proba(original_row.reshape(1, -1))[0]

    cf_pred = classifier.predict(cf_row.reshape(1, -1))[0]
    cf_prob = classifier.predict_proba(cf_row.reshape(1, -1))[0]

    # Step 4: Readable counterfactual
    cf_readable = inverse_transform_row(cf_row, scaler)

    # Step 5: What changed
    changes = {}
    for key in FEATURE_COLS:
        orig_val = round(original_readable[key], 2)
        cf_val   = round(cf_readable[key], 2)
        if abs(orig_val - cf_val) > 0.01:
            changes[key] = {'from': orig_val, 'to': cf_val}

    # Step 6: Smart suggestions — find what WILL flip the outcome
    suggestions = find_suggestions(original_row, original_readable,
                                   orig_pred, scaler, classifier)

    return {
        'original': {
            'values':     {k: round(v, 2) for k, v in original_readable.items()},
            'prediction': int(orig_pred),
            'label':      'Heart Disease' if orig_pred == 1 else 'No Heart Disease',
            'confidence': round(float(max(orig_prob)) * 100, 2)
        },
        'counterfactual': {
            'values':          {k: round(v, 2) for k, v in cf_readable.items()},
            'changed_feature': changed_feature,
            'new_value':       new_value,
            'prediction':      int(cf_pred),
            'label':           'Heart Disease' if cf_pred == 1 else 'No Heart Disease',
            'confidence':      round(float(max(cf_prob)) * 100, 2)
        },
        'changes':         changes,
        'outcome_changed': bool(orig_pred != cf_pred),
        'suggestions':     suggestions
    }


def find_suggestions(original_row, original_readable, orig_pred,
                     scaler, classifier):
    """
    Try changing each feature across its range.
    Return the ones that actually flip the outcome.
    """

    # Define what values to try for each feature
    FEATURE_TRIES = {
        'age':      [30, 40, 50, 60, 70],
        'trestbps': [90, 110, 130, 150, 170],
        'chol':     [150, 200, 250, 300, 350],
        'thalach':  [90, 110, 130, 150, 170],
        'oldpeak':  [0, 0.5, 1.0, 2.0, 3.0, 4.0],
        'sex':      [0, 1],
        'cp':       [0, 1, 2, 3],
        'fbs':      [0, 1],
        'restecg':  [0, 1, 2],
        'exang':    [0, 1],
        'slope':    [0, 1, 2],
        'ca':       [0, 1, 2, 3],
        'thal':     [0, 1, 2, 3]
    }

    suggestions = []

    for feature, values_to_try in FEATURE_TRIES.items():
        for val in values_to_try:
            # Skip if same as current
            if abs(original_readable[feature] - val) < 0.01:
                continue

            # Build a modified row
            test_row = original_row.copy()
            feature_index = FEATURE_COLS.index(feature)

            if feature in NUMERICAL_COLS:
                temp = pd.DataFrame([original_readable])
                temp[feature] = val
                scaled = scaler.transform(temp[NUMERICAL_COLS])[0]
                num_index = NUMERICAL_COLS.index(feature)
                test_row[feature_index] = scaled[num_index]
            else:
                test_row[feature_index] = float(val)

            # Check if outcome flips
            new_pred = classifier.predict(test_row.reshape(1, -1))[0]
            new_prob = classifier.predict_proba(test_row.reshape(1, -1))[0]

            if new_pred != orig_pred:
                suggestions.append({
                    'feature':     feature,
                    'change_to':   val,
                    'from_value':  round(original_readable[feature], 2),
                    'confidence':  round(float(max(new_prob)) * 100, 2),
                    'new_label':   'Heart Disease' if new_pred == 1 else 'No Heart Disease'
                })

    # Return top 4 most confident suggestions
    suggestions.sort(key=lambda x: x['confidence'], reverse=True)
    return suggestions[:4]