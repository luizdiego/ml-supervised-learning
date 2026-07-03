"""
=============================================================================
FUNDAMENTAL CONCEPTS - SUPERVISED LEARNING
=============================================================================

This script demonstrates essential concepts:
1. Bias vs Variance (Underfitting vs Overfitting)
2. Cross-Validation
3. Learning Curve
4. Hyperparameter Search
=============================================================================
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
    GridSearchCV,
    validation_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print("=" * 60)
print("  FUNDAMENTAL CONCEPTS")
print("=" * 60)

# Create synthetic dataset for demonstrations
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42,
)

# ─── 1. BIAS vs VARIANCE ─────────────────────────────────────────────────────
print("\n\n📚 1. BIAS vs VARIANCE (Underfitting vs Overfitting)")
print("-" * 60)
print("""
  UNDERFITTING (High Bias):
  - Model is too simple, doesn't learn patterns
  - Poor performance on BOTH train and test
  - Fix: more complex model, more features

  OVERFITTING (High Variance):
  - Model "memorizes" training data (even the noise!)
  - Great on train, poor on test
  - Fix: more data, regularization, simpler model

  THE GOAL: balance between bias and variance
""")

# Practical demonstration
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("  Decision Tree demonstration:")
print(f"  {'Depth':<15} {'Train Acc':<15} {'Test Acc':<15} {'Diagnosis'}")
print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

for depth in [1, 3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)

    if train_acc < 0.80:
        diag = "Underfitting ⚠️"
    elif train_acc - test_acc > 0.15:
        diag = "Overfitting ⚠️"
    else:
        diag = "Good balance ✓"

    depth_str = str(depth) if depth else "No limit"
    print(f"  {depth_str:<15} {train_acc:<15.4f} {test_acc:<15.4f} {diag}")

# ─── 2. CROSS-VALIDATION ─────────────────────────────────────────────────────
print("\n\n📚 2. CROSS-VALIDATION")
print("-" * 60)
print("""
  Why use it?
  - A single train/test split can be "lucky"
  - Cross-validation splits data into K parts (folds)
  - Trains K times, each time using a different fold as test
  - Result: mean +/- standard deviation of performance

  K-Fold (K=5):
  [Train][Train][Train][Train][TEST ]  -> score 1
  [Train][Train][Train][TEST ][Train]  -> score 2
  [Train][Train][TEST ][Train][Train]  -> score 3
  [Train][TEST ][Train][Train][Train]  -> score 4
  [TEST ][Train][Train][Train][Train]  -> score 5
""")

model = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"  Random Forest with 5-Fold Cross-Validation:")
print(f"  Scores: {scores.round(4)}")
print(f"  Mean:   {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"\n  -> More reliable result than a single split!")

# ─── 3. LEARNING CURVE ───────────────────────────────────────────────────────
print("\n\n📚 3. LEARNING CURVE")
print("-" * 60)
print("""
  Shows how performance changes with the amount of training data.
  Useful to know if more data would help.
""")

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    cv=5,
    scoring='accuracy',
)

print(f"  {'% Data':<12} {'Train Acc':<15} {'Test Acc':<15} {'Gap'}")
print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*10}")

for size, train_s, test_s in zip(train_sizes, train_scores, test_scores):
    pct = size / len(X) * 100
    t_mean = train_s.mean()
    v_mean = test_s.mean()
    gap = t_mean - v_mean
    print(f"  {pct:>5.0f}%      {t_mean:<15.4f} {v_mean:<15.4f} {gap:.4f}")

print("\n  -> If the gap decreases with more data, collecting more data helps!")
print("  -> If both converge to a low value, the model needs to be more complex")

# ─── 4. HYPERPARAMETER SEARCH ────────────────────────────────────────────────
print("\n\n📚 4. HYPERPARAMETER SEARCH (GridSearchCV)")
print("-" * 60)
print("""
  Hyperparameters: model settings that are NOT learned from data
  Examples: n_estimators, max_depth, learning_rate

  GridSearchCV: tests all possible combinations + cross-validation
""")

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 10],
}

print(f"  Testing {3*3} hyperparameter combinations...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
)
grid_search.fit(X, y)

print(f"\n  Best hyperparameters: {grid_search.best_params_}")
print(f"  Best score (CV):      {grid_search.best_score_:.4f}")

print("\n  Top 3 combinations:")
results = grid_search.cv_results_
indices = np.argsort(results['rank_test_score'])[:3]
for i, idx in enumerate(indices, 1):
    params = results['params'][idx]
    score = results['mean_test_score'][idx]
    print(f"    {i}. {params} -> {score:.4f}")

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  📋 SUMMARY - SUPERVISED ML WORKFLOW")
print("=" * 60)
print("""
  1. 📊 Collect and explore data
     - Understand distributions, missing values, correlations

  2. 🧹 Preprocess
     - Handle missing values, categorical encoding, normalization

  3. ✂️  Split data (train/validation/test)
     - NEVER use test data during training!

  4. 🤖 Train models (start simple!)
     - Baseline -> more complex models

  5. 🔧 Optimize hyperparameters
     - GridSearchCV or RandomizedSearchCV

  6. 📈 Evaluate on TEST set
     - Appropriate metrics for the problem

  7. 🚀 Deploy (if satisfied with performance)
     - Save model with joblib/pickle
""")
