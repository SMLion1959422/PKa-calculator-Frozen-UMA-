"""Full traceback, plus a check on whether _tag_and_reparse returns the
atom we actually asked for."""
import sys, traceback
import numpy as np
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from umapka import PkaPredictor
import umapka.predictor as up

smi = "OC(=O)CC(N)C(=O)O"
nm = up.neutralize(Chem.MolFromSmiles(smi))
print(f"neutralized: {Chem.MolToSmiles(nm)}")
print(f"atoms: {[(a.GetIdx(), a.GetSymbol()) for a in nm.GetAtoms()]}\n")

print("=== does _tag_and_reparse preserve the requested atom? ===")
for idx in [0, 5, 8]:
    if idx >= nm.GetNumAtoms():
        continue
    want = nm.GetAtomWithIdx(idx).GetSymbol()
    s, ni = up._tag_and_reparse(nm, idx)
    if s is None:
        print(f"  asked {idx} ({want}): FAILED"); continue
    got = Chem.MolFromSmiles(s).GetAtomWithIdx(ni).GetSymbol()
    flag = "OK" if got == want else "*** ELEMENT MISMATCH ***"
    print(f"  asked idx={idx} ({want}) -> smiles={s}  new_idx={ni} ({got})  {flag}")

print("\n=== full traceback from state_features_v4 ===")
p = PkaPredictor("models/model_core_v3.pkl")
prot, pi_ = up._tag_and_reparse(nm, 0)
try:
    g, l = p.state_features_v4(prot, pi_, "acid", n_confs_base=1)
    print(f"  OK: global={g.shape} local={l.shape}")
except Exception:
    traceback.print_exc()

print("\n=== does the plain (non-v4) path work? ===")
try:
    a, si, mh = up._smiles_to_atoms_with_site(prot, pi_)
    emb = p.embeddings(a)
    print(f"  embeddings OK: {emb.shape}")
    print(f"  pool: {p.pool(emb).shape}")
    print(f"  pool_local: {p.pool_local(emb, si, mh).shape}")
    try:
        print(f"  pool_local_multiscale: {p.pool_local_multiscale(emb, si, mh).shape}")
    except Exception:
        traceback.print_exc()
except Exception:
    traceback.print_exc()
