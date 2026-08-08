import sys, sklearn, os
if "venv311" not in sys.prefix:
    sys.exit(f"WRONG PYTHON: {sys.prefix}\n  activate venv311 first: .\\venv311\\Scripts\\Activate.ps1")

"""POLYPROTIC / MICROSTATE PREDICTION - adopts stages A and D of the
Uni-pKa workflow (microstate enumeration + species distribution) using
the existing v16 model.

WHAT THIS ADDS
  - enumerates EVERY ionizable site, not just the first SMARTS match
  - predicts a micro-pKa for each site independently
  - assembles the sequential deprotonation ladder
  - reports species distribution vs pH, including fraction charged at
    physiological pH 7.4

WHAT IT IS NOT
  Uni-pKa predicts a free energy per microstate and combines them
  thermodynamically, which enforces consistency between coupled sites.
  We predict each site independently, which is the standard
  "non-interacting sites" approximation. It is good when sites are
  well separated (>2-3 pKa units) and degrades when two ionizable
  groups are close enough to influence each other electronically.
  Implementing true FE2pKa needs a differentiable head (torch MLP)
  trained end-to-end - a rewrite, not an addition.

USAGE
  python predict_polyprotic.py "OC(=O)CC(N)C(=O)O"
  python predict_polyprotic.py "CC(=O)Oc1ccccc1C(=O)O" --ph 7.4
"""
import sys, argparse
import numpy as np
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka import PkaPredictor
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)

MODEL = "models/model_core_v16_elec.pkl"

def elec_desc(smi, site_idx):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or site_idx is None or site_idx >= mol.GetNumAtoms(): return None
    try: AllChem.ComputeGasteigerCharges(mol)
    except Exception: return None
    q = np.nan_to_num(np.array([float(a.GetPropsAsDict().get("_GasteigerCharge",0.0))
                                 for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    try: est = np.array(EStateIndices(mol))
    except Exception: est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[site_idx] <= 1)[0]; s2 = np.where(dm[site_idx] <= 2)[0]
    s3 = np.where(dm[site_idx] <= 3)[0]; a = mol.GetAtomWithIdx(site_idx)
    return np.array([q[site_idx], est[site_idx],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()), float(a.GetFormalCharge()),
        float(a.GetIsAromatic()), float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol), float(Chem.GetFormalCharge(mol))],
        dtype=float)

def all_sites(mol):
    """Every ionizable site - ALL matches of ALL patterns, deduplicated
    by atom index (stage A: microstate enumeration)."""
    out, seen = [], set()
    for name, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is None: continue
        for m in mol.GetSubstructMatches(pt):
            idx = m[ai]
            if idx not in seen:
                seen.add(idx); out.append((name, "acid", idx))
    for name, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is None: continue
        for m in mol.GetSubstructMatches(pt):
            idx = m[ai]
            if idx not in seen:
                seen.add(idx); out.append((name, "base", idx))
    return out

def species_distribution(pkas, ph):
    """Fraction of each protonation state at a given pH.
    pkas ascending = sequential deprotonation ladder."""
    n = len(pkas)
    logs = [0.0]
    run = 0.0
    for pk in pkas:
        run += (ph - pk)
        logs.append(run)
    mx = max(logs)
    w = [10.0 ** (l - mx) for l in logs]
    tot = sum(w)
    return [x / tot for x in w]

ap = argparse.ArgumentParser()
ap.add_argument("smiles")
ap.add_argument("--ph", type=float, default=7.4)
ap.add_argument("--model", default=MODEL)
args = ap.parse_args()

b = joblib.load(args.model)
gbm, ridge, sc, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

mol = Chem.MolFromSmiles(args.smiles)
if mol is None: raise SystemExit(f"could not parse: {args.smiles}")
nm = neutralize(mol)
sites = all_sites(nm)
if not sites: raise SystemExit("no ionizable sites found")

print(f"\nMolecule : {args.smiles}")
print(f"Found {len(sites)} candidate ionizable site(s)\n")
print(f"{'site':22s} {'atom':>5s} {'type':6s} {'pKa':>7s}")
print("-" * 45)

results = []
for name, kind, idx in sites:
    try:
        if kind == "acid":
            prot, pi_ = _tag_and_reparse(nm, idx)
            dep, di_ = _shift_hydrogen_tagged(nm, idx, -1, -1)
        else:
            dep, di_ = _tag_and_reparse(nm, idx)
            prot, pi_ = _shift_hydrogen_tagged(nm, idx, +1, +1)
        if prot is None or dep is None:
            print(f"{name:22s} {idx:5d} {kind:6s}   (pair build failed)"); continue
        hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
        hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
        g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        dp = elec_desc(prot, pi_); dd = elec_desc(dep, di_)
        if dp is None or dd is None:
            print(f"{name:22s} {idx:5d} {kind:6s}   (descriptor failed)"); continue
        feat = np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)
        raw = (1-bw)*gbm.predict(feat)[0] + bw*ridge.predict(sc.transform(feat))[0]
        pka = float(cal.predict([raw])[0])
        results.append({"name": name, "kind": kind, "atom": idx, "pKa": pka})
        print(f"{name:22s} {idx:5d} {kind:6s} {pka:7.2f}")
    except Exception as e:
        print(f"{name:22s} {idx:5d} {kind:6s}   (failed: {type(e).__name__})")

if not results: raise SystemExit("\nno sites could be predicted")

pkas = sorted(r["pKa"] for r in results)
n_base = sum(1 for r in results if r["kind"] == "base")

print(f"\n--- DEPROTONATION LADDER ---")
for i, pk in enumerate(pkas, 1):
    print(f"  pKa{i} = {pk:.2f}")

print(f"\n--- SPECIES DISTRIBUTION AT pH {args.ph} ---")
frac = species_distribution(pkas, args.ph)
for j, fr in enumerate(frac):
    chg = n_base - j
    label = f"charge {chg:+d}" if chg else "neutral"
    bar = "#" * int(round(fr * 40))
    print(f"  {label:12s} {fr*100:6.2f}%  {bar}")

neutral_idx = n_base
frac_charged = 1.0 - frac[neutral_idx] if 0 <= neutral_idx < len(frac) else 1.0
print(f"\n  fraction CHARGED at pH {args.ph}: {frac_charged*100:.1f}%")
print("  (drives membrane permeability, hERG binding, BBB penetration)")

print(f"\n--- pH PROFILE ---")
print(f"{'pH':>5s}  " + "  ".join(f"q{n_base-j:+d}" .rjust(6) for j in range(len(pkas)+1)))
for ph in [1,2,3,4,5,6,7,7.4,8,9,10,11,12,13]:
    fr = species_distribution(pkas, ph)
    print(f"{ph:5.1f}  " + "  ".join(f"{x*100:5.1f}%" for x in fr))

print("""
NOTE: sites are predicted INDEPENDENTLY (non-interacting approximation).
Reliable when sites are >2-3 pKa units apart; degrades for strongly
coupled ionizable groups. Uni-pKa's thermodynamic microstate treatment
handles coupling properly but needs a differentiable end-to-end head.""")

