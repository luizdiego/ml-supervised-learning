"""
=============================================================================
SUPERVISED LEARNING - CLASSIFICATION
=============================================================================

What is Classification?
- Predict the CATEGORY (class) of a new data point
- Examples: spam/not-spam, sick/healthy, cat/dog

In this example: Classify Iris flowers into 3 species based on
petal and sepal measurements.

Typical pipeline:
1. Load data
2. Explore and understand the data
3. Split into train/test
4. Train model
5. Evaluate performance
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("=" * 60)
print("  CLASSIFICATION - Iris Dataset")
print("=" * 60)

# ─── STEP 1: Load data ───────────────────────────────────────────────────────
iris = load_iris()
X = iris.data          # Features: flower measurements
y = iris.target        # Labels: flower species (0, 1, or 2)

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

print("\n📊 STEP 1: Understanding the data")
print(f"   Total samples: {len(df)}")
print(f"   Features (inputs): {iris.feature_names}")
print(f"   Classes (outputs): {list(iris.target_names)}")
print(f"\n   First 5 samples:")
print(df.head().to_string(index=False))

# ─── STEP 2: Split into train and test ───────────────────────────────────────
# Why split? To evaluate if the model generalizes well to NEW data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% for test, 70% for training
    random_state=42,    # Reproducibility
    stratify=y          # Maintain class proportions
)

print("\n\n📐 STEP 2: Train/test split")
print(f"   Training data: {len(X_train)} samples")
print(f"   Test data:     {len(X_test)} samples")

# ─── STEP 3: Feature Scaling ─────────────────────────────────────────────────
# Why scale? Distance-based algorithms (like KNN) are sensitive to feature scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on training
X_test_scaled = scaler.transform(X_test)         # only transform on test!

print("\n\n⚖️  STEP 3: Feature Scaling")
print(f"   Before - mean: {X_train[:, 0].mean():.2f}, std: {X_train[:, 0].std():.2f}")
print(f"   After  - mean: {X_train_scaled[:, 0].mean():.2f}, std: {X_train_scaled[:, 0].std():.2f}")

# ─── STEP 4: Train models ────────────────────────────────────────────────────
print("\n\n🤖 STEP 4: Model Training")

# Model 1: K-Nearest Neighbors (KNN)
# Idea: classify based on the K closest neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print("   ✓ KNN (K=5) trained")

# Model 2: Decision Tree
# Idea: create if/else rules to separate classes
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train_scaled, y_train)
print("   ✓ Decision Tree (depth=3) trained")

# ─── STEP 5: Evaluate models ─────────────────────────────────────────────────
print("\n\n📈 STEP 5: Evaluation on TEST set")

for name, model in [("KNN", knn), ("Decision Tree", tree)]:
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n   --- {name} ---")
    print(f"   Accuracy: {acc:.2%}")
    print(f"   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   {cm}")

# ─── STEP 6: Predict on new data ─────────────────────────────────────────────
print("\n\n🔮 STEP 6: Prediction on new data")
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Measurements of an unknown flower
new_flower_scaled = scaler.transform(new_flower)

prediction = knn.predict(new_flower_scaled)
probabilities = knn.predict_proba(new_flower_scaled)

print(f"   Flower measurements: {new_flower[0]}")
print(f"   Prediction: {iris.target_names[prediction[0]]}")
print(f"   Probabilities per class:")
for name, prob in zip(iris.target_names, probabilities[0]):
    print(f"     {name}: {prob:.2%}")

# ─── KEY CONCEPTS ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  💡 KEY CLASSIFICATION CONCEPTS")
print("=" * 60)
print("""
  • Accuracy: % of correct predictions (be careful with imbalanced data!)
  • Precision: of those the model predicted as X, how many actually are X?
  • Recall: of those that actually are X, how many did the model find?
  • F1-Score: harmonic mean between precision and recall
  • Overfitting: model memorizes training data, fails on test
  • Underfitting: model is too simple, doesn't learn patterns
""")
