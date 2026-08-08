"""MAXIMUM-DATA RUN: v16's winning recipe (UMA + Gasteiger + hybrid head)
applied to every molecule we can site-assign defensibly.

Three sources:
  1. marvin_atom molecules  (~5,184) - ground-truth sites, current v16
  2. extra_pka_data         (~8,491) - SMARTS sites, kept ONLY where the
                                        site is chemically consistent with
                                        the label (the v7-clean filter)
  3. hunt_et_al             (~1,763) - explicit atom-mapped sites, if built

UMA features already exist for all of these in feat_train_v6.pkl. What
is missing is electronic descriptors for sources 2/3, which this
computes (RDKit only, ~1 min).

This is the last untried lever. Expected landing zone 0.80-0.83 on
Novartis, not 0.75 - stated up front so the result is judged against an
honest prior."""
import sys, numpy as np, pandas as pd, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import (ACID_SITES, BASE_SITES, neutralize,
                               _tag_and_reparse, _shift_hydrogen_tagged)

SITE_RANGES = {
    "carboxylic_acid": (1.5,6.5), "sulfonic_acid": (-2.0,3.0),
    "phosphoric_acid": (0.5,4.0), "tetrazole": (3.0,7.0),
    "tetrazole_2": (3.0,7.0), "sulfonamide_2": (6.5,12.5),
    "sulfonamide_1": (8.0,13.0), "thiol": (6.5,12.0),
    "hydroxamic_acid": (7.0,11.5), "phenol": (6.5,12.5),
    "imide": (6.0,12.0), "aromatic_lactam": (7.5,13.0),
    "malonate_dicarbonyl": (3.0,13.0), "cyanoacetate": (7.0,13.0),
    "malononitrile": (8.0,13.0), "oxime": (8.5,13.0),
    "guanidine": (10.0,14.0), "amidine": (8.5,13.0),
    "prim_amine": (7.5,12.0), "sec_amine": (7.5,12.0),
    "tert_amine": (6.5,11.5), "pyridine_N": (2.0,8.0),
    "aniline": (1.5,6.5), "aniline_sec": (1.5,7.0), "aniline_tert": (2.0,7.5),
}

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

def named_site(mol):
    for n_, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return n_, "acid", m[0][ai]
    for n_, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return n_, "base", m[0][ai]
    return None, None, None

print("loading caches...")
f6 = joblib.load("feat_train_v6.pkl")
valid6 = {s for s,v in f6.items() if np.asarray(v).shape == (2304,)}
elec_old = joblib.load("feat_electronic.pkl")
corrected = joblib.load("feat_marvin_corrected.pkl")
try: hunt = joblib.load("feat_hunt.pkl")
except FileNotFoundError: hunt = {}; print("  (hunt not built - skipping source 3)")
print(f"  UMA={len(valid6)}  elec={len(elec_old)}  corrected={len(corrected)}  hunt={len(hunt)}")

rows, seen = [], set()

# --- source 1: marvin_atom molecules ---
n1 = 0
for mol in Chem.ForwardSDMolSupplier("mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
    except Exception: continue
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms() or smi not in elec_old: continue
    _, _, pidx = named_site(nm)
    vec = None
    if pidx is not None and pidx == ma and smi in valid6: vec = f6[smi]
    elif smi in corrected: vec = corrected[smi]["feat"]; exp = corrected[smi]["pKa"]
    if vec is None or smi in seen: continue
    seen.add(smi); n1 += 1
    rows.append({"smiles": smi, "pKa": exp, "vec": np.concatenate([vec, elec_old[smi]])})
print(f"\nsource 1 (marvin sites): {n1}")

# --- source 2: extra_pka_data, plausibility-filtered ---
n2 = n2_drop = 0
try:
    extra = pd.read_csv("extra_pka_data.csv")
    for r in tqdm(extra.itertuples(), total=len(extra), desc="source 2"):
        smi = r.smiles
        if smi in seen or smi not in valid6: continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        nm = neutralize(mol)
        name, kind, idx = named_site(nm)
        if name is None: continue
        rng = SITE_RANGES.get(name)
        if rng is None or not (rng[0] <= r.pKa <= rng[1]):
            n2_drop += 1; continue          # implausible -> drop (v7-clean filter)
        try:
            if kind == "acid":
                prot, pi_ = _tag_and_reparse(nm, idx); dep, di_ = _shift_hydrogen_tagged(nm, idx, -1, -1)
            else:
                dep, di_ = _tag_and_reparse(nm, idx); prot, pi_ = _shift_hydrogen_tagged(nm, idx, +1, +1)
            if prot is None or dep is None: continue
            dp = elec_desc(prot, pi_); dd = elec_desc(dep, di_)
            if dp is None or dd is None: continue
        except Exception: continue
        seen.add(smi); n2 += 1
        rows.append({"smiles": smi, "pKa": float(r.pKa),
                     "vec": np.concatenate([f6[smi], dp, dd, dp - dd])})
except FileNotFoundError:
    print("  extra_pka_data.csv not found")
print(f"source 2 (extra, plausible): {n2}   dropped implausible: {n2_drop}")

# --- source 3: hunt_et_al ---
n3 = 0
if hunt:
    try:
        hp = pd.read_csv("hunt_pairs.csv").set_index("key")
        for k, v in tqdm(hunt.items(), desc="source 3"):
            if k in seen or np.asarray(v["feat"]).shape != (2304,): continue
            if k not in hp.index: continue
            row = hp.loc[k]
            dp = elec_desc(row.prot_smi, int(row.prot_idx))
            dd = elec_desc(row.dep_smi, int(row.dep_idx))
            if dp is None or dd is None: continue
            seen.add(k); n3 += 1
            rows.append({"smiles": k, "pKa": v["pKa"],
                         "vec": np.concatenate([v["feat"], dp, dd, dp - dd])})
    except FileNotFoundError:
        print("  hunt_pairs.csv missing")
print(f"source 3 (hunt): {n3}")

core = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
print(f"\nTOTAL training set: {len(core)}   (v16 used 5184 -> {(len(core)/5184-1)*100:+.0f}%)")
joblib.dump(core, "core_maxdata.pkl")
print("saved -> core_maxdata.pkl")
