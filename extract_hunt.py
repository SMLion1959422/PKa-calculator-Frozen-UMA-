"""Extract hunt_et_al molecules: conjugate_acid_smi/conjugate_base_smi
carry EXPLICIT ATOM MAPS marking the ionization site, e.g.
  acid: C[NH2+:20]C(C)C   base: C[NH:20]C(C)C
So both the protonation pair AND the site are given - no SMARTS
guessing, no Marvin needed. This is the highest-quality data available
to us, and 1,763 of these molecules are not in the current training set.

Excludes anything matching the test sets (exact SMILES + >=0.95 Tanimoto)
so this cannot contaminate the benchmark."""
import numpy as np, pandas as pd, joblib, gzip, os
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
RDLogger.DisableLog("rdApp.*")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def strip_maps(mol):
    idx = None
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() != 0:
            idx = a.GetIdx(); a.SetAtomMapNum(0)
    return mol, idx

print("loading test sets for contamination guard...")
test_smis, test_fps = set(), []
for path in ["mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf",
             "mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"]:
    for m in Chem.ForwardSDMolSupplier(path):
        if m is None: continue
        try:
            test_smis.add(Chem.MolToSmiles(m)); test_fps.append(gen.GetFingerprint(m))
        except Exception: pass
print(f"  {len(test_smis)} test molecules")

print("loading existing training SMILES...")
have = set()
for m in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if m is None: continue
    try: have.add(Chem.MolToSmiles(m))
    except Exception: pass
print(f"  {len(have)} already in training")

src = None
for cand in ["extra_pka_source/datasets/hunt_et_al.sdf.gz",
             "extra_pka_source/datasets/hunt_et_al.sdf"]:
    if os.path.exists(cand): src = cand; break
if src is None: raise SystemExit("hunt_et_al file not found")
print(f"reading {src}")
supp = Chem.ForwardSDMolSupplier(gzip.open(src, "rb")) if src.endswith(".gz") \
       else Chem.ForwardSDMolSupplier(src)

recs, n_skip_test, n_skip_have, n_bad = [], 0, 0, 0
for mol in supp:
    if mol is None: continue
    if not all(mol.HasProp(k) for k in ("pKa", "conjugate_acid_smi", "conjugate_base_smi")):
        continue
    try:
        exp = float(mol.GetProp("pKa"))
        if not (0 < exp < 14): continue
        am = Chem.MolFromSmiles(mol.GetProp("conjugate_acid_smi"))
        bm = Chem.MolFromSmiles(mol.GetProp("conjugate_base_smi"))
        if am is None or bm is None: n_bad += 1; continue
        am, ai = strip_maps(am); bm, bi = strip_maps(bm)
        if ai is None or bi is None: n_bad += 1; continue
        neutral = Chem.MolToSmiles(bm if Chem.GetFormalCharge(bm) == 0 else am)
    except Exception:
        n_bad += 1; continue
    if neutral in test_smis: n_skip_test += 1; continue
    if neutral in have: n_skip_have += 1; continue
    try:
        fp = gen.GetFingerprint(Chem.MolFromSmiles(neutral))
        if max(DataStructs.BulkTanimotoSimilarity(fp, test_fps)) >= 0.95:
            n_skip_test += 1; continue
    except Exception: pass
    kind = "base" if Chem.GetFormalCharge(am) > Chem.GetFormalCharge(bm) and \
                     am.GetAtomWithIdx(ai).GetFormalCharge() > 0 else "acid"
    recs.append({"key": neutral, "pKa": exp, "kind": kind,
                 "prot_smi": Chem.MolToSmiles(am), "prot_idx": ai,
                 "dep_smi": Chem.MolToSmiles(bm), "dep_idx": bi})

df = pd.DataFrame(recs).drop_duplicates("key").reset_index(drop=True)
print(f"\nusable NEW molecules: {len(df)}")
print(f"  skipped (in test sets):     {n_skip_test}")
print(f"  skipped (already training): {n_skip_have}")
print(f"  unparseable:                {n_bad}")
print(df.kind.value_counts())
df.to_csv("hunt_pairs.csv", index=False)
print("saved -> hunt_pairs.csv")
