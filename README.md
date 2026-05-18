# Credit Score Classification

Data Science Term Project - team 6

## 1. Project Overview

This project aims to build a machine learning model that predicts a customer's credit score class.

- Dataset: Credit Score Classification
- Source: Kaggle
- Task: Multi-class Classification
- Target: `Credit_Score`
- Classes: `Good`, `Standard`, `Poor`

The project follows an end-to-end machine learning workflow:

1. Data exploration
2. Data preprocessing
3. Model training
4. Model evaluation
5. Experiment tracking with MLflow

## 2. Project Structure

```text
credit-score-project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_test.ipynb
│   └── 03_model_test.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── train.py
```

## 3. Virtual Environment Setup

Before running the project, create a virtual environment and install the required packages.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Run Training Script

```bash
python3 src/train.py
```

1. Load raw data
2. Clean and preprocess data
3. Split train/validation data
4. Train model
5. Evaluate model
