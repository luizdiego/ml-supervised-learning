# 🤖 Supervised Learning - Practical Guide

## What is Supervised Learning?

A type of Machine Learning where we have **labeled data** (input -> expected output pairs).
The model learns to map inputs to outputs and generalizes to new data.

## Project Structure

```
ml-supervised-learning/
├── 01_classification.py   # Classification with Iris dataset
├── 02_regression.py       # Regression with California Housing
├── 03_fundamentals.py     # Bias/Variance, Cross-Val, GridSearch
└── README.md              # This file
```

## How to Run

```bash
# Prerequisites
pip install scikit-learn pandas numpy matplotlib

# Run each script
python 01_classification.py
python 02_regression.py
python 03_fundamentals.py
```

## Contents

### 01 - Classification
- Iris dataset (flowers)
- KNN and Decision Tree
- Accuracy, Confusion Matrix
- Prediction on new data

### 02 - Regression
- California Housing dataset (house prices)
- Linear Regression, Ridge, Decision Tree, Random Forest
- MAE, RMSE, R2
- Coefficient interpretation

### 03 - Fundamentals
- Bias vs Variance (Under/Overfitting)
- Cross-Validation (K-Fold)
- Learning Curve
- Hyperparameter Search (GridSearchCV)

## Next Steps

- 📚 Explore more algorithms: SVM, Gradient Boosting, Neural Networks
- 📊 Work with real-world datasets (Kaggle)
- 🔧 Learn feature engineering
- 🚀 Deploy with Flask/FastAPI
