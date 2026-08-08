import pandas as pd
import glob

for fn in glob.glob('*.csv'):
    try:
        df = pd.read_csv(fn)
        if 'dataset' in df.columns and 'true_pka' in df.columns:
            print(f'--- Scanning {fn} ---')
            sub = df[(df['dataset'] == 'novartis') & (df['size'] > 22)].copy()
            sub['err'] = abs(sub['pred_pka'] - sub['true_pka'])
            print(sub.sort_values('err', ascending=False)[['smiles', 'true_pka', 'pred_pka', 'err', 'size']].head(10))
    except Exception as e:
        pass
