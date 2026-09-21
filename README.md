# Streamlit Model App

An interactive Streamlit application for exploring a tuberculosis burden dataset and training simple classification models to predict **Region** from user-selected features.

## Project Scope and Purpose

This project provides a lightweight, UI-driven workflow for:
- previewing a tabular dataset,
- selecting input features,
- training/evaluating classifiers,
- comparing model and feature-level performance.

It is aimed at rapid experimentation and educational model evaluation rather than production model serving.

## High-Level Architecture

The app is organized into two main Python modules:

- **`app.py`**
  - Builds the Streamlit interface (sidebar controls, dataset preview, model/feature comparison views).
  - Loads the cleaned dataset from GitHub (`TB_Burden_Country_Cleaned.xlsx`) using `requests` + `pandas.read_excel`.
  - Manages persistent run state with `st.session_state`.
- **`model_functions.py`**
  - Implements model training and evaluation helpers:
    - Gaussian Naive Bayes
    - k-Nearest Neighbors (with optional best-`k` search)
  - Supports both train/test split and 10-fold cross-validation.
  - Renders model/feature comparison tables and charts.

### Runtime Flow

1. App loads dataset.
2. User selects features in the sidebar (target is fixed to `Region`).
3. User chooses classifier and evaluation method.
4. App trains/evaluates and stores results in session state.
5. User can compare multiple models and per-feature runs in tabular and chart form.

## Technology Stack

Based on `requirements.txt` and source code:

- **Python 3.11** (dev container base image)
- **Streamlit** (`streamlit==1.34.0`) for web UI
- **pandas / numpy** for data handling
- **scikit-learn** for ML models and evaluation
- **matplotlib / seaborn** for visualizations
- **openpyxl** for Excel file support
- **requests** for fetching dataset content from GitHub

## Key Features

- Interactive feature selection from dataset columns
- Classifier selection:
  - Gaussian Naive Bayes
  - kNN (manual `k` or automatic search)
- Evaluation methods:
  - Train-test split
  - 10-fold cross-validation
- Session-persistent tracking of multiple model runs
- Model comparison table + accuracy bar chart
- Individual feature evaluation and comparison

## Usage

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal (typically `http://localhost:8501`).

## Configuration, Data, and Deployment Notes

- The app currently fetches the cleaned Excel dataset from:
  - `https://raw.githubusercontent.com/ejvluna/streamlit-model-app/main/TB_Burden_Country_Cleaned.xlsx`
- Local dataset files (`TB_Burden_Country*.xlsx/.csv`) are present in the repository, but the runtime path in `app.py` uses the remote URL.
- `.devcontainer/devcontainer.json` includes a ready-to-run Streamlit setup and auto-start command for containerized development.
