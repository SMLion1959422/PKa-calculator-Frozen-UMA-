"""Verify marvin_atom annotations before using them. Checks indexing
convention, which element it points at, and how often it disagrees with
the current SMARTS priority pick (= how much error it could fix)."""
import sys
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

def priority_pick(mol):
    for name, sm, ai in ACID_SITES:
        p = Chem.MolFromSmarts(sm)
        if p is not None:
            m = mol.GetSubstructMatches(p)
            if m:
                return name, "acid", m[0][ai]
    for name, sm, ai in BASE_SITES:
        p = Chem.MolFromSmarts(sm)
        if p is not None:
            m = mol.GetSubstructMatches(p)
            if m:
                return name, "base", m[0][ai]
    return None, None, None

for path, label in [
    ("mlpka/datasets/combined_training_datasets_unique.sdf", "TRAINING"),
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "NOVARTIS"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "AVLILUMOVE"),
]:
    print("=" * 62)
    print(f"{label}: {path}")
    print("=" * 62)
    supp = Chem.ForwardSDMolSupplier(path)
    elem0, elem1 = Counter(), Counter()
    types = Counter()
    n = n_bad0 = n_bad1 = 0
    agree_atom = agree_kind = comparable = 0
    for mol in supp:
        if mol is None or not mol.HasProp("marvin_atom"):
            continue
        try:
            ma = int(float(mol.GetProp("marvin_atom")))
        except Exception:
            continue
        n += 1
        na = mol.GetNumAtoms()
        if 0 <= ma < na:
            elem0[mol.GetAtomWithIdx(ma).GetSymbol()] += 1
        else:
            n_bad0 += 1
        if 0 <= ma - 1 < na:
            elem1[mol.GetAtomWithIdx(ma - 1).GetSymbol()] += 1
        else:
            n_bad1 += 1
        if mol.HasProp("marvin_pKa_type"):
            types[mol.GetProp("marvin_pKa_type")] += 1
        try:
            nm = neutralize(Chem.Mol(mol))
            pname, pkind, pidx = priority_pick(nm)
            if pidx is not None:
                comparable += 1
                if pidx == ma:
                    agree_atom += 1
                mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
                mk = "acid" if mt.startswith("acid") else "base" if mt.startswith("basic") else None
                if mk and mk == pkind:
                    agree_kind += 1
        except Exception:
            pass
        if n >= 3000:
            break

    print(f"molecules with marvin_atom: {n}")
    print(f"\n0-BASED  element at marvin_atom (out-of-range: {n_bad0}):")
    print(f"  {dict(elem0.most_common(8))}")
    print(f"1-BASED  element at marvin_atom-1 (out-of-range: {n_bad1}):")
    print(f"  {dict(elem1.most_common(8))}")
    print(f"\n  >>> the convention whose elements are mostly N/O/S is correct")
    print(f"\nmarvin_pKa_type: {dict(types)}")
    if comparable:
        print(f"\nvs current SMARTS priority pick (n={comparable}):")
        print(f"  same ATOM index:   {agree_atom} ({agree_atom/comparable*100:.1f}%)")
        print(f"  same ACID/BASE:    {agree_kind} ({agree_kind/comparable*100:.1f}%)")
        print(f"  >>> disagreement = molecules currently featurized wrong")
    print()
