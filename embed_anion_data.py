import numpy as np, pandas as pd, joblib, time, os
from rdkit import Chem
from umapka import PkaPredictor
from umapka.predictor import _smiles_to_atoms

p = PkaPredictor("models/model_core.pkl")
df = pd.read_csv(r"anion_data\data\D2A-pKa.csv")

# split reaction_smiles "AH>>A-" into (protonated, deprotonated)
def split_rxn(rxn):
    try:
        a, b = rxn.split(">>")
        return a.strip(), b.strip()
    except: return None, None

# resume from checkpoint if it exists
cache_path = "solvent_feat_cache.pkl"
feat_cache = joblib.load(cache_path) if os.path.exists(cache_path) else {}
print(f"starting from {len(feat_cache)} cached embeddings")

# collect all unique molecule SMILES needing embedding
needed = set()
for rxn in df["reaction_smiles"]:
    a, b = split_rxn(rxn)
    if a and b:
        needed.add(a); needed.add(b)
needed -= set(feat_cache.keys())
print(f"molecules to embed: {len(needed)}")

t0 = time.time()
items = list(needed)
for i, smi in enumerate(items):
    try:
        feat_cache[smi] = p.pool(p.embeddings(_smiles_to_atoms(smi)))
    except Exception:
        feat_cache[smi] = None
    if (i+1) % 300 == 0:
        el = time.time()-t0
        rem = (el/(i+1))*(len(items)-i-1)
        print(f"[{i+1}/{len(items)}] {el/60:.1f}min elapsed, ~{rem/60:.1f}min left")
        joblib.dump(feat_cache, cache_path)

joblib.dump(feat_cache, cache_path)
ok = sum(1 for v in feat_cache.values() if v is not None)
print(f"done. {ok} valid embeddings cached -> {cache_path}")
