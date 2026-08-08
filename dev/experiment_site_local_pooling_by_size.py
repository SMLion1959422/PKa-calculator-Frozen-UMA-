"""
Follow-up to experiment_site_local_pooling.py: breaks the same
global-vs-local comparison down by molecule size, using the same
buckets as RESULTS.md's own size-dependence table. This tells you
WHERE any improvement (or lack of one) is coming from, not just
whether the overall average moved.

Run from the repo root, venv311 active:
    python dev/experiment_site_local_pooling_by_size.py
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem

from umapka import PkaPredictor
from umapka.site_features import (
    protonation_pair_with_site, smiles_to_atoms_with_site, pool_site, self_test,
)

print("=== sanity check: does site tagging land on the right atom? ===")
self_test()
print()

p = PkaPredictor("models/model_core.pkl")

df = pd.read_csv(r"anion_data\data\D2A-pKa.csv")
df = df[df["solvent_smiles"] == "O"].reset_index(drop=True)
print(f"water rows: {len(df)}")

def split_rxn(r):
    try:
        a, b = r.split(">>")
        return a.strip(), b.strip()
    except Exception:
        return None, None

X_global, X_both, y, n_heavy = [], [], [], []
t0 = time.time()
n_skipped = 0
for i, row in df.iterrows():
    a, b = split_rxn(row["reaction_smiles"])
    if not a or not b:
        n_skipped += 1
        continue
    try:
        mol_a = Chem.MolFromSmiles(a)
        if mol_a is None:
            n_skipped += 1
            continue
        heavy = mol_a.GetNumAtoms()

        prot, deprot = protonation_pair_with_site(a)
        atoms_p, site_p = smiles_to_atoms_with_site(prot)
        atoms_d, site_d = smiles_to_atoms_with_site(deprot)
        if site_p is None or site_d is None:
            n_skipped += 1
            continue
        emb_p = p.embeddings(atoms_p)
        emb_d = p.embeddings(atoms_d)
        h_p_global = p.pool(emb_p)
        h_d_global = p.pool(emb_d)
        h_p_local = pool_site(emb_p, atoms_p.get_positions(), site_p)
        h_d_local = pool_site(emb_d, atoms_d.get_positions(), site_d)

        feat_global = np.concatenate(
            [h_p_global, h_d_global, h_p_global - h_d_global])
        h_p_both = np.concatenate([h_p_global, h_p_local])
        h_d_both = np.concatenate([h_d_global, h_d_local])
        feat_both = np.concatenate([h_p_both, h_d_both, h_p_both - h_d_both])
    except Exception:
        n_skipped += 1
        continue
    X_global.append(feat_global)
    X_both.append(feat_both)
    y.append(row["pKa_avg"])
    n_heavy.append(heavy)
    if (i + 1) % 200 == 0:
        el = time.time() - t0
        print(f"  [{i+1}/{len(df)}]  {el/60:.1f} min elapsed")

print(f"usable: {len(y)}  (skipped {n_skipped})")
X_global = np.array(X_global)
X_both = np.array(X_both)
y = np.array(y)
n_heavy = np.array(n_heavy)

def mk():
    return lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              learning_rate=0.05, verbose=-1, random_state=42)

kf = KFold(5, shuffle=True, random_state=42)

def cv_predictions(X):
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        m = mk()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred

pred_global = cv_predictions(X_global)
pred_both = cv_predictions(X_both)

bins = [0, 15, 22, 30, 999]
labels = ["<15", "15-22", "22-30", ">30"]
bucket = pd.cut(n_heavy, bins=bins, labels=labels, right=False)

print(f"\n{'size (heavy atoms)':<20}{'n':>6}{'global MAE':>12}{'+local MAE':>12}{'delta':>10}")
for lbl in labels:
    mask = bucket == lbl
    if mask.sum() == 0:
        continue
    mae_g = np.abs(pred_global[mask] - y[mask]).mean()
    mae_b = np.abs(pred_both[mask] - y[mask]).mean()
    print(f"{lbl:<20}{mask.sum():>6}{mae_g:>12.3f}{mae_b:>12.3f}{mae_b-mae_g:>+10.3f}")

mae_g_all = np.abs(pred_global - y).mean()
mae_b_all = np.abs(pred_both - y).mean()
print(f"\n{'OVERALL':<20}{len(y):>6}{mae_g_all:>12.3f}{mae_b_all:>12.3f}{mae_b_all-mae_g_all:>+10.3f}")
