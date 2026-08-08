"""Dump all data fields present in the source SDFs - looking for
pre-computed protonation-site annotations (atom indices, site labels,
acid/base flags) that we may have discarded when we extracted only
(SMILES, pKa) pairs."""
import glob, os
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

paths = sorted(glob.glob("extra_pka_source/datasets/*.sdf*")) + \
        sorted(glob.glob("mlpka/datasets/*.sdf"))

for path in paths:
    try:
        if path.endswith(".gz"):
            import gzip
            fh = gzip.open(path, "rb")
            supp = Chem.ForwardSDMolSupplier(fh)
        else:
            supp = Chem.ForwardSDMolSupplier(path)
        mol = None
        for m in supp:
            if m is not None:
                mol = m
                break
        if mol is None:
            print(f"\n{os.path.basename(path)}: no readable molecule")
            continue
        props = list(mol.GetPropNames())
        print(f"\n{os.path.basename(path)}")
        print(f"  fields: {props}")
        for p in props:
            v = str(mol.GetProp(p))
            print(f"    {p} = {v[:70]}")
    except Exception as e:
        print(f"\n{os.path.basename(path)}: ERROR {e}")

print("\n\nLOOKING FOR: any field naming an atom index, site type,")
print("acid/base flag, or 'reaction center' - that would be real")
print("ground-truth site annotation instead of my heuristic ranges.")
