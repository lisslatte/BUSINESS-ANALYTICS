# Foundational Business Analytics Coursework (2025–2026)
**Restaurant Inspection Risk Prediction System**

This folder contains a fully functional Python model pipeline to predict whether a restaurant inspection will result in a high-risk score (SCORE >= 14).

---
## 📁 Contents
- `fba_pipeline.py` — Full training and feature-engineering pipeline.
- `predict_new.py` — Script for predicting high-risk inspections on new data.
- `final_model.joblib` — Trained RandomForest model.
- `preprocessor.joblib` — Preprocessing pipeline for numeric and categorical transformations.
- `README.md` — This guide.

---
## ⚙️ Requirements
Python 3.9+ and the following libraries:
```bash
pip install pandas numpy scikit-learn joblib
```

---
## 🚀 How to Run

### 1. Train the model (optional)
Ensure your training CSV (`FBA Coursework Data.csv`) is in the same directory.
```bash
python fba_pipeline.py
```
This will generate and save new model artifacts (`final_model.joblib`, `preprocessor.joblib`).

### 2. Predict on new inspections
Provide a CSV file with the same schema (without the `SCORE` column).
```bash
python predict_new.py new_data.csv
```
Output file: `predicted_output.csv`

This file includes:
- `Risk_Probability`: model’s estimated probability that the inspection is high-risk
- `Predicted_High_Risk`: 1 = high-risk, 0 = low-risk

---
## 🧠 Model Overview
- Model: **Random Forest Classifier**
- Evaluation: Chronological time-split validation
- Tuned to achieve **Recall ≈ 0.90** with **Precision ≈ 0.91**
- Avoids data leakage by using only information from *prior inspections* for each restaurant.

---
## 🏁 Notes
- The pipeline automatically engineers key historical and trend-based features.
- Designed to meet the business objective of identifying high-risk restaurants **before** a health crisis occurs.
