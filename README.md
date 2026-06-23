# Diabetes Prediction App

A machine learning web application that predicts the likelihood of diabetes based on patient health metrics. Built with Logistic Regression and deployed via Streamlit.

## Project Structure

```
DiabetesPredict/
├── diabetes.py                     # Streamlit web app
├── diabetes_lr.pkl                 # Trained model + scaler (joblib)
├── Legit_diabetes.ipynb            # Model training notebook
├── Legit_diabetes_Test.ipynb       # Testing/validation notebook
├── FinalProj_DiabetesPredict.pdf   # Project report
├── requirements.txt                # Python dependencies
└── .devcontainer/
    └── devcontainer.json           # VS Code / Codespaces config
```

## Features

- **Input**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Output**: "Likely to have Diabetes" / "Unlikely to have Diabetes"
- Models evaluated: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes
- Best model (Logistic Regression) saved with `joblib`

## Quick Start

```bash
pip install -r requirements.txt
streamlit run diabetes.py
```

Or open in GitHub Codespaces — the devcontainer auto-launches the app on port 8501.

## Dataset

Uses the Pima Indians Diabetes Database. Features are standardized with `StandardScaler` before prediction.

## Tech Stack

- Python, pandas, scikit-learn
- Streamlit (frontend)
- Jupyter Notebooks (training/analysis)
