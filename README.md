# Explainable ML Web App for Multi-Level PMOS Risk Assessment

A web-based clinical decision support system enabling decentralized, at-home risk screening and longitudinal symptom tracking for **Polyendocrine Metabolic Ovarian Syndrome (PMOS)** using **Explainable AI (SHAP)**.

## Problem & Solution
Patients frequently experience delayed preventive care or unnecessary hospital visits due to limited access to transparent pre-clinical screening tools. This application combines a multi-level Random Forest risk stratification pipeline with **SHAP (SHapley Additive exPlanations)** so users can inspect which specific metabolic and endocrine biomarkers drove their risk tier.

## Key Features
- **Multi-Level Risk Stratification:** Evaluates 8 endocrine, metabolic, and ovulatory markers (BMI, HOMA-IR, Fasting Glucose, LH/FSH ratio, Free Testosterone, Cycle Variance, Triglycerides, and mFG Hirsutism Score) into Low, Moderate, and High risk tiers.
- **SHAP Feature Attribution Visuals:** Uses `shap.TreeExplainer` to decompose each prediction into per-feature risk drivers and protective factors, resolving the AI "black box" problem for non-technical users.
- **At-Home Symptom & Screening Log:** Tracks check-in history over time to support proactive lifestyle adjustments before clinical consultation.

## Tech Stack
- **Backend & ML Pipeline:** Python, Django, Scikit-learn, SHAP, NumPy
- **Frontend:** React 18, JavaScript (ES6+), HTML5/CSS3

## Local Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Start the Django server:
   ```bash
   python manage.py runserver
   ```
3. Open `http://127.0.0.1:8000` in your browser.