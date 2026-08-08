import pandas as pd

df = pd.read_csv('characterization_external_v19.csv')
sub = df[(df['dataset'] == 'novartis') & (df['n_atoms'] > 22)].copy()

cols_to_show = [c for c in ['exp', 'pred', 'err', 'n_atoms', 'site_correct', 'pka_bin'] if c in sub.columns]
print(sub.sort_values('err', ascending=False)[cols_to_show].head(10))
