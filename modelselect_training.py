import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


"""
Cody Smith
Model Selection and Training 
April 10th, 2026
"""

""" 
this code WILL need alterations once all data cleaning and preprocessing is done

cleaned data file read in:
    df = pd.read_csv('(cleaned_data_file.csv')

Other alterations:
    Feature importance handoff will need to be integrated into final complete code 

Save trained models with:
    joblib.dump(lr_model, 'lr_model.pk1')
    (James can then load directly for final eval with no retraining)

n_estimators and max_depth can be done in a loop (like HW2) to find
the best Random Forest configuration before next steps.
"""

""" 
Assuming the data is cleaned, the df with a Z-score is NOT yet applied 
"""

np.random.seed(42)

X = df.drop(columns=['finishing_time'])
y = df['finishing_time'].values

" 80/20 split "
X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
)

" StandardScaler, fit to the training data only "
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

" Model 1 (Linear Regression) "
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_train_pred_lr = lr_model.predict(X_train_scaled)
y_val_pred_lr   = lr_model.predict(X_val_scaled)

train_mae_lr = mean_absolute_error(y_train, y_val_pred_lr)
val_mae_lr   = mean_absolute_error(y_val,   y_val_pred_lr)

print(f"Linear Regression - Train MAE: {train_mae_lr:.2f} min | Val MAE: {val_mae_lr:.2f} min")

" Model 2 (Random Forest) "
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_train_pred_rf = rf_model.predict(X_train_scaled)
y_val_pred_rf   = rf_model.predict(X_val_scaled)

train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
val_mae_rf   = mean_absolute_error(y_val,   y_val_pred_rf)

print(f"Random Forest   - Train MAE: {train_mae_rf:.2f} min | Val MAE: {val_mae_rf:.2f} min")

" Comparison DataFrame "
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest (100 trees)'],
    'Train MAE (min)': [train_mae_lr, train_mae_rf],
    'Val MAE (min)':   [val_mae_lr,   val_mae_rf]
})

print("\n" + "="*55)
print("MODEL COMPARISON")
print("="*55)
print(comparison_df.to_string(index=False))

" Feature importance from the Random Forest "
feature_names       = list(X.columns)
importances         = rf_model.feature_importances_
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

" Model Performance/Prediction Plotting "
plt.figure(figsize=(10, 5))
feature_importances.plot(kind='bar', color='steelblue', edgecolor='black')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Random Forest - Feature Importances (Marathon)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

" Feature selection display "
print("\nTop 5 most important features:")
print(feature_importances.head(5).to_string())





