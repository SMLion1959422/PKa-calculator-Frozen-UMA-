"""Find the real AttributeError - no broad exception swallowing."""
import sys, traceback
import numpy as np
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from umapka import PkaPredictor
import umapka.predictor as up

print("=== does state_features_v4 exist? ===")
print("PkaPredictor methods containing 'features':")
print(" ", [m for m in dir(PkaPredictor) if "features" in m.lower()])
print("\nPkaPredictor methods containing 'pool':")
print(" ", [m for m in dir(PkaPredictor) if "pool" in m.lower()])
print("\nmodule-level helpers:")
print(" ", [m for m in dir(up) if m.startswith("_smiles") or m.startswith("_shift") or m.startswith("_tag")])
print(f"\npredictor.py loaded from: {up.__file__}")

print("\n=== live call with FULL traceback ===")
p = PkaPredictor("models/model_core_v3.pkl")
nm = up.neutralize(Chem.MolFromSmiles("OC(=O)CC(N)C(=O)O"))
prot, pi_ = up._tag_and_reparse(nm, 0)
dep, di_ = up._shift_hydrogen_tagged(nm, 0, -1, -1)
print(f"prot={prot} idx={pi_}")
print(f"dep ={dep} idx={di_}")
try:
    r = p.state_features_v4(prot, pi_, "acid", n_confs_base=1)
    print(f"state_features_v4 OK -> shapes {r[0].shape}, {r[1].shape}")
except Exception:
    traceback.print_exc()
