import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""
James Kurschner
Evaluation Logic and Documentation
April 23, 2026
"""

# Load saved models
lr_model = joblib.load('lr_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Previous validation split 
df = pd.read_csv('csv/train_processed.csv')
X = df.drop(columns=['actual_finish_time_minutes'])
y = df['actual_finish_time_minutes'].values
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline prediction: Computes average for a baseline of comparison
baseline_pred = np.full(len(y_val), np.mean(y_train))

# Validation predictions
y_val_pred_lr = lr_model.predict(X_val)
y_val_pred_rf = rf_model.predict(X_val)

# Compute Mean Absolute Error
baseline_mae = mean_absolute_error(y_val, baseline_pred)
lr_mae = mean_absolute_error(y_val, y_val_pred_lr)
rf_mae = mean_absolute_error(y_val, y_val_pred_rf)

# Compute Root Mean Squared Error
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
lr_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lr))
rf_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))

# Compute Coefficient of Determination
baseline_r2 = r2_score(y_val, baseline_pred)
lr_r2 = r2_score(y_val, y_val_pred_lr)
rf_r2 = r2_score(y_val, y_val_pred_rf)

# Comparison Dataframe
comparison_df = pd.DataFrame({
    'Model': ['Baseline (Mean)', 'Linear Regression', 'Random Forest'],
    'Val MAE (min)': [baseline_mae, lr_mae, rf_mae],
    'Val RMSE (min)': [baseline_rmse, lr_rmse, rf_rmse],
    'Val R^2': [baseline_r2, lr_r2, rf_r2]
})

print('\n' + '=' * 60)
print('Model Comparison')
print('=' * 60)
print(comparison_df.to_string(index=False))

# Actual vs predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val_pred_lr, alpha=0.5, label='Linear Regression')
plt.scatter(y_val, y_val_pred_rf, alpha=0.5, label='Random Forest')

min_val = min(y_val.min(), y_val_pred_lr.min(), y_val_pred_rf.min())
max_val = max(y_val.max(), y_val_pred_lr.max(), y_val_pred_rf.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

plt.xlabel('Actual Finish Time (min)')
plt.ylabel('Predicted Finish Time (min)')
plt.title('Actual vs Predicted Marathon Finish Times')
plt.legend()
plt.tight_layout()
plt.savefig('results/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
