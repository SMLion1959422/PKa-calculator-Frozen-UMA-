"""Isolate the AttributeError - test elec_desc alone, no UMA, with the
traceback exposed instead of swallowed."""
import sys, traceback
import numpy as np
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
from umapka.predictor import neutralize, _tag_and_reparse

nm = neutralize(Chem.MolFromSmiles("OC(=O)CC(N)C(=O)O"))
prot, pi_ = _tag_and_reparse(nm, 0)
print(f"testing on: {prot}  site_idx={pi_}\n")

mol = Chem.MolFromSmiles(prot)

print("step 1: ComputeGasteigerCharges")
try:
    AllChem.ComputeGasteigerCharges(mol); print("  OK")
except Exception: traceback.print_exc()

print("\nstep 2: read charges via GetPropsAsDict")
try:
    q = [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0)) for a in mol.GetAtoms()]
    print(f"  OK: {np.round(q,3)}")
except Exception:
    traceback.print_exc()
    print("\n  trying GetDoubleProp fallback:")
    try:
        q = [float(a.GetDoubleProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
             for a in mol.GetAtoms()]
        print(f"  FALLBACK OK: {np.round(q,3)}")
    except Exception: traceback.print_exc()

print("\nstep 3: EStateIndices")
try:
    est = np.array(EStateIndices(mol)); print(f"  OK: shape {est.shape}")
except Exception: traceback.print_exc()

print("\nstep 4: descriptors")
for name, fn in [("TPSA", Descriptors.TPSA), ("MolLogP", Crippen.MolLogP)]:
    try: print(f"  {name}: {fn(mol):.2f}")
    except Exception: traceback.print_exc()

print("\nstep 5: atom accessors")
try:
    a = mol.GetAtomWithIdx(pi_)
    for m in ["GetDegree","GetTotalNumHs","GetFormalCharge","GetIsAromatic","IsInRing","GetAtomicNum"]:
        print(f"  {m}: {getattr(a,m)()}")
except Exception: traceback.print_exc()
