import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

"""
John Cordwell III
Feature Selection
April 27, 2026
"""

# load processed dataset and split features from target
df = pd.read_csv("csv/train_processed.csv")
X = df.drop(columns=["actual_finish_time_minutes"])
y = df["actual_finish_time_minutes"]

# training / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = joblib.load("rf_model.pkl")

perm = permutation_importance(
    rf_model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

# store and sort highest to lowest
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv("csv/permutation_feature_importance.csv", index=False)

# top 10 features
top_features = importance_df.head(10)["feature"].tolist()

plt.figure(figsize=(10, 6))
plt.barh(
    importance_df.head(10)["feature"][::-1],
    importance_df.head(10)["importance_mean"][::-1]
)
plt.xlabel("Permutation Importance")
plt.title("Top 10 Features by Permutation Importance")
plt.tight_layout()
plt.savefig("results/permutation_feature_importance.png", dpi=150)


models = {
    "Linear regression (All Features)": LinearRegression(),
    "Random Forest (All Features)": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear regression (Top 10 Features)": LinearRegression(),
    "Random Forest (Top 10 Features)": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = []

for name, model in models.items():
    if "Top 10" in name:
        Xtr = X_train[top_features]
        Xv = X_val[top_features]
    else:
        Xtr = X_train
        Xv = X_val

    model.fit(Xtr, y_train)
    preds = model.predict(Xv)

    results.append({
        "model": name,
        "features_used": len(Xtr.columns),
        "MAE": mean_absolute_error(y_val, preds),
        "R2": r2_score(y_val, preds)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("csv/feature_selection_model_comparison.csv", index=False)

print("\nTop 10 selected features:")
print(top_features)

print("\nFeature selection model comparison:")
print(results_df.to_string(index=False))
