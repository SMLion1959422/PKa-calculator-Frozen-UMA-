"""Embed the hunt_et_al pairs using their given atom-mapped sites."""
import numpy as np, pandas as pd, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from umapka import PkaPredictor

OUT, PARTIAL = "feat_hunt.pkl", "feat_hunt.pkl.partial"
df = pd.read_csv("hunt_pairs.csv")
print(f"{len(df)} molecules to embed")
try:
    out = joblib.load(PARTIAL); print(f"resuming: {len(out)}")
except FileNotFoundError: out = {}

print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")
todo = df[~df.key.isin(out.keys())]
print(f"{len(todo)} remaining\n")
n_fail = 0
for r in tqdm(todo.itertuples(), total=len(todo)):
    try:
        hg_p, hl_p = p.state_features_v4(r.prot_smi, int(r.prot_idx), r.kind, n_confs_base=1)
        hg_d, hl_d = p.state_features_v4(r.dep_smi, int(r.dep_idx), r.kind, n_confs_base=1)
        g = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        l = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        out[r.key] = {"feat": np.concatenate([g, l]), "pKa": r.pKa, "kind": r.kind}
    except Exception:
        n_fail += 1
    if len(out) % 50 == 0: joblib.dump(out, PARTIAL)
joblib.dump(out, OUT)
print(f"\ndone: {len(out)} embedded, {n_fail} failed -> {OUT}")
