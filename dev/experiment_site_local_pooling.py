"""
Controlled A/B test: does adding site-local pooled features reduce
water-pKa MAE compared to the existing global-mean+max-only features?

Uses the water subset of the anion dataset (same source
dev/train_anion_fast.py trains on). Both feature variants are built
from the SAME 3D-embedded conformer and the SAME UMA embeddings per
molecule - the only difference is pooling strategy - so the delta
between the two MAE numbers below isolates the effect of local
pooling, controlling for everything else.

Caveat: this re-derives the deprotonation site itself via SMARTS
(protonation_pair_with_site), rather than using the dataset's exact
paired SMILES like embed_anion_data.py does. That means the "global
pooling only" baseline number below may differ slightly from the
documented 0.64 MAE - that documented number used the dataset's exact
pairs. Compare the two numbers THIS script prints to each other, not
to 0.64.

Run from the repo root, venv311 active:
    python dev/experiment_site_local_pooling.py
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import KFold
import lightgbm as lgb

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

X_global, X_both, y = [], [], []
t0 = time.time()
n_skipped = 0
for i, row in df.iterrows():
    a, b = split_rxn(row["reaction_smiles"])
    if not a or not b:
        n_skipped += 1
        continue
    try:
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
    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f"  [{i+1}/{len(df)}]  {el/60:.1f} min elapsed")

print(f"usable: {len(y)}  (skipped {n_skipped})")
X_global = np.array(X_global)
X_both = np.array(X_both)
y = np.array(y)

def mk():
    return lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              learning_rate=0.05, verbose=-1, random_state=42)

kf = KFold(5, shuffle=True, random_state=42)

def cv_mae(X):
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        m = mk()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return np.abs(pred - y).mean()

mae_global = cv_mae(X_global)
mae_both = cv_mae(X_both)

print(f"\n{'features':<32}{'5-fold MAE':>12}")
print(f"{'global pooling only (current)':<32}{mae_global:>12.3f}")
print(f"{'global + site-local pooling':<32}{mae_both:>12.3f}")
print(f"\ndelta: {mae_both - mae_global:+.3f}  "
      f"({'better' if mae_both < mae_global else 'worse or no change'})")
