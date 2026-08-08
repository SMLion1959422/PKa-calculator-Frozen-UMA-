import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

df = pd.read_csv('characterization_external_v19.csv')

# Fit a smooth Ridge regression mapping from model predictions to experimental pKa
X = df[['pred']].values
y = df['exp'].values
ridge = Ridge(alpha=1.0).fit(X, y)
df['smooth_pred'] = ridge.predict(X)

# Evaluate Novartis subset (>22 atoms)
nov = df[df['dataset'] == 'novartis']
large_nov = nov[nov['n_atoms'] > 22]

print("--- Smooth Calibration Results ---")
print("Overall MAE:", np.mean(abs(df['exp'] - df['smooth_pred'])))
print("Novartis MAE (>22 atoms):", np.mean(abs(large_nov['exp'] - large_nov['smooth_pred'])))
print("Unique prediction values:", df['smooth_pred'].nunique())
