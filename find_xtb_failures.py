"""WHICH molecules did xTB fail on? The charge hypothesis is dead (all
12 test cases passed, ions included), so this finds the real pattern by
comparing the full input list against what actually made it into
feat_xtb.pkl.

RUN IN THE 'xtb' CONDA ENV. Takes ~1 minute (no xTB calls - it only
reads the saved results and re-derives the input list).

Then it re-runs xTB on a sample of the FAILURES with full error
reporting, so we see the actual exception rather than guessing.
"""
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from xtb.interface import Calculator
from xtb.utils import get_method
import sys

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import (neutralize, _tag_and_reparse,
                               _shift_hydrogen_tagged)

BOHR = 1.8897259886

print("loading feat_xtb.pkl...")
done = joblib.load("feat_xtb.pkl")
print(f"  {len(done)} succeeded")

print("rebuilding the full input list...")
targets = []
for path in ["mlpka/datasets/combined_training_datasets_unique.sdf",
             "mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf",
             "mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"]:
    for mol in Chem.ForwardSDMolSupplier(path):
        if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
            continue
        try:
            exp = float(mol.GetProp("pKa"))
            ma = int(float(mol.GetProp("marvin_atom")))
            smi = Chem.MolToSmiles(mol)
            nm = neutralize(Chem.Mol(mol))
        except Exception:
            continue
        if not (0 < exp < 14) or ma >= nm.GetNumAtoms():
            continue
        mt = mol.GetProp("marvin_pKa_type") if mol.HasProp("marvin_pKa_type") else ""
        targets.append((smi, ma, "acid" if mt.startswith("acid") else "base"))

seen, uniq = set(), []
for t in targets:
    if t[0] not in seen:
        seen.add(t[0])
        uniq.append(t)

failed = [t for t in uniq if t[0] not in done]
succeeded = [t for t in uniq if t[0] in done]
print(f"  {len(uniq)} unique | {len(succeeded)} ok | {len(failed)} failed "
      f"({len(failed)/len(uniq)*100:.1f}%)")


def profile(rows, label):
    recs = []
    for smi, ma, kind in rows:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        elems = {a.GetSymbol() for a in m.GetAtoms()}
        recs.append({
            "n_atoms": m.GetNumAtoms(),
            "n_heavy": m.GetNumHeavyAtoms(),
            "mw": Descriptors.MolWt(m),
            "rings": Descriptors.RingCount(m),
            "rotb": Descriptors.NumRotatableBonds(m),
            "kind": kind,
            "elems": elems,
        })
    df = pd.DataFrame(recs)
    print(f"\n--- {label} (n={len(df)}) ---")
    print(df[["n_atoms", "n_heavy", "mw", "rings", "rotb"]].describe().round(1).loc[
        ["mean", "50%", "max"]])
    print(f"  acid/base: {dict(df.kind.value_counts())}")
    ec = Counter()
    for s in df.elems:
        ec.update(s)
    print(f"  elements: {dict(ec.most_common(12))}")
    return df

df_ok = profile(succeeded, "SUCCEEDED")
df_bad = profile(failed, "FAILED")

print("\n" + "=" * 62)
print("KEY COMPARISON")
print("=" * 62)
for col in ["n_atoms", "n_heavy", "mw", "rings", "rotb"]:
    print(f"  {col:9s}  ok={df_ok[col].mean():7.1f}   failed={df_bad[col].mean():7.1f}"
          f"   ratio={df_bad[col].mean()/max(df_ok[col].mean(),1e-9):.2f}x")

ok_el = Counter()
for s in df_ok.elems:
    ok_el.update(s)
bad_el = Counter()
for s in df_bad.elems:
    bad_el.update(s)
only_bad = set(bad_el) - set(ok_el)
if only_bad:
    print(f"\n  elements appearing ONLY in failures: {sorted(only_bad)}")

print("\n" + "=" * 62)
print("RE-RUNNING xTB ON 15 FAILURES WITH FULL ERRORS")
print("=" * 62)
errs = Counter()
for smi, ma, kind in failed[:15]:
    try:
        nm = neutralize(Chem.MolFromSmiles(smi))
        if kind == "acid":
            prot, pi_ = _tag_and_reparse(nm, ma)
            dep, di_ = _shift_hydrogen_tagged(nm, ma, -1, -1)
        else:
            dep, di_ = _tag_and_reparse(nm, ma)
            prot, pi_ = _shift_hydrogen_tagged(nm, ma, +1, +1)
        if prot is None or dep is None:
            print(f"  PAIR BUILD FAILED: {smi[:55]}")
            errs["pair_build"] += 1
            continue
        for tag, s, idx in (("prot", prot, pi_), ("deprot", dep, di_)):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                errs[f"{tag}_parse"] += 1
                continue
            charge = Chem.GetFormalCharge(mol)
            molh = Chem.AddHs(mol)
            p = AllChem.ETKDGv3()
            p.randomSeed = 42
            if AllChem.EmbedMolecule(molh, p) != 0:
                print(f"  EMBED FAILED ({tag}, {molh.GetNumAtoms()} atoms): {s[:50]}")
                errs[f"{tag}_embed"] += 1
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molh)
            except Exception:
                pass
            nums = np.array([a.GetAtomicNum() for a in molh.GetAtoms()])
            pos = molh.GetConformer().GetPositions() * BOHR
            try:
                c = Calculator(get_method("GFN2-xTB"), nums, pos, charge=float(charge))
                c.set_verbosity(0)
                c.singlepoint()
                errs[f"{tag}_ok"] += 1
            except Exception as e:
                print(f"  XTB FAILED ({tag}, {molh.GetNumAtoms()} atoms, q={charge:+d}): "
                      f"{type(e).__name__}: {str(e)[:60]}")
                errs[f"{tag}_xtb"] += 1
    except Exception as e:
        errs[f"outer_{type(e).__name__}"] += 1

print(f"\n  {dict(errs)}")
print("""
READ THIS AS:
  *_embed dominant  -> RDKit 3D conformer generation is the bottleneck,
                       not xTB. Would be fixed by more embedding attempts
                       or useRandomCoords=True.
  *_xtb dominant    -> genuine xTB SCF failures on real drug molecules.
  pair_build        -> our own protonation-pair construction failing.

Either way, remember xTB ALREADY LOST to free Gasteiger descriptors
(0.541 vs 0.521) on the molecules where it DID work - so this is about
reporting the result honestly, not about rescuing xTB.""")
