# backend/train.py
# Run this ONE TIME to train and save both models
# After this you never need to train again!

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from backend.preprocess import load_data, fit_scaler, transform
from backend.model import train_classifier
from backend.vae import train_vae

print("=" * 40)
print("Step 1: Loading data...")
X, y, df = load_data()
print(f"Loaded {len(df)} rows")

print("\nStep 2: Fitting scaler...")
scaler = fit_scaler(X)

print("\nStep 3: Scaling features...")
X_scaled = transform(X, scaler)

print("\nStep 4: Training classifier...")
clf = train_classifier(X_scaled, y)

print("\nStep 5: Training VAE...")
vae = train_vae(X_scaled, epochs=50)

print("\n" + "=" * 40)
print("All models saved in backend/models/")
print("You can now run app.py!")