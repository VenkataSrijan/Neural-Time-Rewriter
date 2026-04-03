# 🧠 Neural Time Rewriter
### A Counterfactual Deep Learning System for Cardiac Decision Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

**Neural Time Rewriter** is a counterfactual deep learning system that goes
beyond traditional prediction — it answers the critical question:

> *"What would have happened if things were different?"*

Given a patient's clinical data, the system:
1. Predicts whether the patient is at risk of heart disease
2. Explains **why** that prediction was made
3. Simulates **what-if scenarios** by modifying individual features
4. Suggests which specific changes would flip the outcome

This makes AI predictions **transparent, explainable, and actionable** —
which is essential in medical decision support systems.

---

## 👥 Team

| Name | Role |
|------|------|
| Srijan | Model Architecture, VAE Design |
| Ashvith | Backend API, Data Pipeline |
| Bharath | Frontend UI, Integration |

---

## 🏥 Dataset

**UCI Heart Disease Dataset — Cleveland Clinic Foundation**

- **Source:** UCI Machine Learning Repository
- **Collected by:** Robert Detrano, M.D., Ph.D. — Cleveland Clinic Foundation
- **Published in:** *American Journal of Cardiology*, 1988
- **Patients:** 303 real patients
- **Features:** 13 clinical features
- **Citation:** Detrano, R., et al. (1989). International application of a new
  probability algorithm for the diagnosis of coronary artery disease.
  *American Journal of Cardiology*, 64(5), 304–310.

This dataset has been used in **1,000+ published research papers** and is
the gold standard benchmark for cardiac ML research.

### Feature Description

| Feature | Description | Normal Range |
|---------|-------------|--------------|
| `age` | Age in years | 29–77 |
| `sex` | Sex (1=Male, 0=Female) | — |
| `cp` | Chest pain type (0–3) | — |
| `trestbps` | Resting blood pressure | 90–120 mmHg |
| `chol` | Serum cholesterol | <200 mg/dl |
| `fbs` | Fasting blood sugar >120 | 0 or 1 |
| `restecg` | Resting ECG results (0–2) | — |
| `thalach` | Maximum heart rate achieved | 60–200 bpm |
| `exang` | Exercise-induced angina | 0 or 1 |
| `oldpeak` | ST depression (exercise vs rest) | 0–2.0 |
| `slope` | Slope of peak exercise ST | 0–2 |
| `ca` | Major vessels (fluoroscopy) | 0–3 |
| `thal` | Thalassemia type | 0–3 |
| `target` | **Heart disease present** | **0 or 1** |

---

## 🏗️ System Architecture

```
User Input
│
▼
Frontend (HTML/CSS/JS)
│  REST API calls
▼
Flask Backend (app.py)
│
├──► Prediction Model (RandomForest)
│         └── Returns: label + confidence
│
├──► Counterfactual Generator
│         ├── Modifies selected feature
│         ├── VAE smoothing (vae.py)
│         └── Smart suggestions engine
│
└──► Response to Frontend
└── Original vs Counterfactual comparison
```
---

## 🧬 Technical Architecture

### 1. Prediction Model
- **Algorithm:** Random Forest Classifier (100 estimators)
- **Accuracy:** ~87% on held-out test set
- **Why Random Forest:** Handles mixed data types, resistant to overfitting
  on small medical datasets, provides feature importance natively

### 2. Variational Autoencoder (VAE)
- **Purpose:** Learns the statistical distribution of patient data
- **Architecture:** Encoder (13→32→16→6) + Decoder (6→16→32→13)
- **Role in counterfactuals:** Ensures generated patient profiles remain
  realistic and within the learned data distribution

### 3. Counterfactual Engine
- Changes exactly one feature at a time
- Recomputes prediction on the modified profile
- Automatically searches all features to find what WILL flip the outcome
- Returns top 4 most confident suggestions

---

## 📁 Project Structure
```
NNDL project/
│
├── backend/
│   ├── app.py                  # Flask REST API
│   ├── preprocess.py           # Data loading and scaling
│   ├── model.py                # RandomForest classifier
│   ├── vae.py                  # Variational Autoencoder
│   ├── counterfactual.py       # What-if scenario engine
│   ├── train.py                # One-time training script
│   └── models/
│       ├── classifier.pkl      # Saved RandomForest
│       ├── vae.pt              # Saved VAE weights
│       └── scaler.pkl          # Saved StandardScaler
│
├── data/
│   └── heart.csv               # UCI Cleveland Heart Disease dataset
│
├── frontend/
│   ├── index.html              # Main UI
│   ├── style.css               # Dark theme styling
│   └── script.js               # API calls and UI logic
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- VS Code (recommended)
- Live Server extension for VS Code

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/neural-time-rewriter.git
cd neural-time-rewriter
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the models
```bash
python backend/train.py
```
This takes under 60 seconds. Saves 3 files in `backend/models/`.

### Step 5 — Start the Flask server
```bash
python backend/app.py
```
Server runs at `http://127.0.0.1:5000`

### Step 6 — Open the frontend
Open `frontend/index.html` with VS Code Live Server.

---

## 🖥️ How to Use

1. **Load Patient** — Click "Load Random Patient" to load a real patient
   from the dataset, or click "Enter Manually" to input your own values
2. **Predict** — Click "Predict Outcome" to see if the patient is at risk
3. **What If?** — Select any clinical feature and change its value
4. **Counterfactual** — Click "Generate Counterfactual" to see how the
   prediction changes and get smart suggestions for what will flip the outcome

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/features` | GET | Get feature ranges for UI sliders |
| `/api/sample` | GET | Get a random patient from dataset |
| `/api/predict` | POST | Predict outcome for given patient data |
| `/api/counterfactual` | POST | Generate counterfactual scenario |

### Example — Predict
```json
POST /api/predict
{
  "data": {
    "age": 55, "sex": 1, "cp": 2, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2
  }
}
```

### Example — Response
```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 82.5
}
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 86.89% |
| F1 Score | 87.10% |
| Dataset | UCI Cleveland (303 patients) |
| Train/Test Split | 80/20 |
| Cross-validation | Stratified |

> ⚠️ **Medical Disclaimer:** This system is a research prototype built for
> academic purposes. It is trained on a dataset of 303 patients and uses
> statistical patterns — it is **not a substitute for professional medical
> diagnosis**. Always consult a qualified physician for medical advice.

---

## 🔬 What is Counterfactual ML?

Traditional ML models answer: *"What will happen?"*

Counterfactual ML answers: *"What would have happened if X was different?"*

This is critical in medicine because:
- A doctor doesn't just want to know a patient is at risk
- They want to know **what to change** to reduce that risk
- "If this patient's cholesterol drops below 200, their risk drops from
  High to Low" is actionable medical insight

Our system uses a **Variational Autoencoder** to ensure counterfactual
scenarios stay realistic — generated patient profiles always fall within
the distribution of real patient data.

---

## 🚀 Future Enhancements

- [ ] Integration with larger datasets (MIMIC-IV, UK Biobank)
- [ ] SHAP value explanations for each prediction
- [ ] Multi-feature counterfactuals (change 2–3 features at once)
- [ ] Natural language explanations using LLMs
- [ ] Real-time data streaming integration
- [ ] Cloud deployment (AWS/GCP)
- [ ] Mobile responsive UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML Models | Scikit-learn, PyTorch |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 📚 References

1. Detrano, R., et al. (1989). International application of a new probability
   algorithm for the diagnosis of coronary artery disease. *American Journal
   of Cardiology*, 64(5), 304–310.

2. Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual
   Explanations Without Opening the Black Box. *Harvard Journal of Law &
   Technology*, 31(2).

3. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
   *arXiv:1312.6114*.

4. UCI Machine Learning Repository — Heart Disease Dataset.
   https://archive.ics.uci.edu/ml/datasets/Heart+Disease

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file
for details.

---

<p align="center">
  Built by Srijan, Ashvith and Bharath
</p>
