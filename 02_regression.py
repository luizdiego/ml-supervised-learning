"""
=============================================================================
SUPERVISED LEARNING - REGRESSION
=============================================================================

What is Regression?
- Predict a CONTINUOUS (numeric) value
- Examples: house price, tomorrow's temperature, salary

In this example: Predict house prices in California based on
features like location, number of rooms, area income, etc.

Pipeline:
1. Load and explore data
2. Split into train/test
3. Train regression models
4. Evaluate with regression metrics
5. Interpret results
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("  REGRESSION - House Prices (California Housing)")
print("=" * 60)

# ─── STEP 1: Load data ───────────────────────────────────────────────────────
housing = fetch_california_housing()
X = housing.data
y = housing.target  # Median price in hundreds of thousands of dollars

df = pd.DataFrame(X, columns=housing.feature_names)
df['MedianPrice'] = y

print("\n📊 STEP 1: Understanding the data")
print(f"   Total samples: {len(df)}")
print(f"   Features: {housing.feature_names}")
print(f"   Target: Median house price (x $100k)")
print(f"\n   Price statistics:")
print(f"     Min:    ${y.min() * 100_000:,.0f}")
print(f"     Max:    ${y.max() * 100_000:,.0f}")
print(f"     Mean:   ${y.mean() * 100_000:,.0f}")
print(f"     Median: ${np.median(y) * 100_000:,.0f}")

# ─── STEP 2: Split and scale ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n\n📐 STEP 2: Data Split")
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ─── STEP 3: Train models ────────────────────────────────────────────────────
print("\n\n🤖 STEP 3: Training regression models")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })
    print(f"   ✓ {name} trained")

# ─── STEP 4: Compare results ─────────────────────────────────────────────────
print("\n\n📈 STEP 4: Model Comparison")
print("-" * 60)
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print("-" * 60)

for r in results:
    print(f"{r['Model']:<22} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f}")

print("-" * 60)

# Best model
best = max(results, key=lambda x: x['R2'])
print(f"\n   🏆 Best model: {best['Model']} (R2 = {best['R2']:.4f})")

# ─── STEP 5: Interpret coefficients (Linear Regression) ──────────────────────
print("\n\n🔍 STEP 5: Interpretation (Linear Regression)")
lr = models["Linear Regression"]
coefficients = pd.Series(lr.coef_, index=housing.feature_names)
coefficients_sorted = coefficients.abs().sort_values(ascending=False)

print("   Most important features (by coefficient magnitude):")
for feat in coefficients_sorted.index[:5]:
    value = coefficients[feat]
    direction = "↑" if value > 0 else "↓"
    print(f"     {direction} {feat}: {value:+.4f}")

# ─── STEP 6: Predict on new data ─────────────────────────────────────────────
print("\n\n🔮 STEP 6: Prediction on new data")

# Simulate a house: high median income, few people, good location
new_house = np.array([[8.0, 25.0, 6.0, 1.0, 500, 3.0, 37.0, -122.0]])
new_house_scaled = scaler.transform(new_house)

best_model = models[best['Model']]
predicted_price = best_model.predict(new_house_scaled)[0]

print(f"   Features: MedInc=8, HouseAge=25, Rooms=6, ...")
print(f"   Predicted price: ${predicted_price * 100_000:,.0f}")

# ─── KEY CONCEPTS ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  💡 KEY REGRESSION CONCEPTS")
print("=" * 60)
print("""
  Metrics:
  • MAE (Mean Absolute Error): average absolute error — easy to interpret
  • RMSE (Root Mean Squared Error): penalizes large errors
  • R2 (Coefficient of Determination): 1 = perfect, 0 = constant model

  Models:
  • Linear Regression: simple, interpretable, assumes linear relationship
  • Ridge/Lasso: linear regression with regularization (prevents overfitting)
  • Decision Tree: captures non-linearities, but can overfit
  • Random Forest: ensemble of trees, generally more robust

  Tips:
  • Always scale features for distance-based models
  • Negative R2 = model is worse than just predicting the mean!
  • More data usually beats a more complex model
""")
