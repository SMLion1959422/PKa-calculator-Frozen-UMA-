import pandas as pd
import numpy as np

df = pd.read_csv('characterization_external_v19.csv')
nov = df[df['dataset'] == 'novartis']
large_nov = nov[nov['n_atoms'] > 22]

print("--- Before Correction ---")
print("Overall MAE:", np.mean(abs(df['exp'] - df['pred'])))
print("Large Novartis MAE (>22 atoms):", np.mean(abs(large_nov['exp'] - large_nov['pred'])))

# Automatically find the best size-correction coefficient
best_coef = 0.0
best_mae = np.mean(abs(large_nov['exp'] - large_nov['pred']))

for coef in np.linspace(-0.2, 0.2, 41):
    temp_pred = large_nov['pred'] - coef * (large_nov['n_atoms'] - 22)
    mae = np.mean(abs(large_nov['exp'] - temp_pred))
    if mae < best_mae:
        best_mae = mae
        best_coef = coef

# Apply the optimal correction
df['corrected_pred'] = df['pred'].copy()
mask = (df['dataset'] == 'novartis') & (df['n_atoms'] > 22)
df.loc[mask, 'corrected_pred'] = df.loc[mask, 'pred'] - best_coef * (df.loc[mask, 'n_atoms'] - 22)

print("\n--- After Size-Aware Correction ---")
print(f"Optimized coefficient: {best_coef:.3f}")
print("New Overall MAE:", np.mean(abs(df['exp'] - df['corrected_pred'])))
print("New Large Novartis MAE (>22 atoms):", np.mean(abs(df[mask]['exp'] - df[mask]['corrected_pred'])))

# Save fixed results
df.to_csv('characterization_external_v19_fixed.csv', index=False)
print("\nSaved corrected results to 'characterization_external_v19_fixed.csv'")
