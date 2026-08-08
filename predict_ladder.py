"""SEQUENTIAL MICROSTATE LADDER - adopts Uni-pKa's coupled-site logic
without needing a differentiable end-to-end head.

THE BUG THIS FIXES
predict_polyprotic.py predicted every site against the SAME neutral
background, giving aspartic acid's amine pKa 7.46 (true ~9.9). Wrong
because by the time the amine deprotonates, both carboxylates already
carry negative charge, which stabilizes the ammonium and RAISES its
pKa. Independent prediction cannot see that.

THE FIX
Start fully protonated (every base protonated, every acid still H-bearing),
then at each rung:
  1. for EVERY site still bearing a removable H, predict the pKa of
     removing it FROM THE CURRENT CHARGE STATE
  2. the lowest is the next macroscopic pKa
  3. commit that deprotonation, recompute the rest on the new background
So each prediction sees the real electrostatic environment. This is the
"sequential macro-pKa ladder" and it captures site-site coupling through
the features themselves.

HONEST LIMITATION
Uni-pKa predicts a free energy for EVERY microstate and combines them
with a proper partition function, which also captures cases where two
orderings are near-degenerate. We commit greedily to the lowest rung at
each step, so we get one dominant path rather than a full ensemble. We
also step outside the training distribution: the model was trained on
mono-ionizable molecules (mostly neutral<->+/-1), so predicting on a
dianion background is extrapolation. Treat multiply-charged backgrounds
with suspicion and check the flagged warnings below.

USAGE
  python predict_ladder.py "OC(=O)CC(N)C(=O)O"
  python predict_ladder.py "NC(Cc1ccccc1)C(=O)O" --ph 7.4
  python predict_ladder.py "OC(=O)CC(N)C(=O)O" --compare
"""
import sys

import argparse
import numpy as np
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka import PkaPredictor
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize
from umapka import electronic, microstates as M

TAG0 = 101
RENDER_TAG = 99


# elec_desc/all_sites now come from umapka.electronic /
# umapka.microstates - this file used to carry byte-identical
# copies of both, so any fix had to be made in several places.
elec_desc = electronic.elec_desc


def all_sites(mol):
    return M.all_sites(mol, ACID_SITES, BASE_SITES)


def find_tag(mol, tag):
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == tag:
            return a.GetIdx()
    return None


def shift_h(smiles, tag, d_h, d_q):
    """Add/remove one H at the atom carrying `tag`. Keeps ALL tags."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    idx = find_tag(mol, tag)
    if idx is None: return None
    rw = Chem.RWMol(mol)
    a = rw.GetAtomWithIdx(idx)
    n_h = a.GetTotalNumHs() + d_h
    if n_h < 0: return None
    a.SetNumExplicitHs(n_h)
    a.SetNoImplicit(True)
    a.SetFormalCharge(a.GetFormalCharge() + d_q)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return None


def render_site(smiles, tag):
    """Strip all tags except mark `tag`, return (clean_smiles, site_idx)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, None
    idx = find_tag(mol, tag)
    if idx is None: return None, None
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
        # CRITICAL: shift_h() sets NoImplicit(True), which SURVIVES the
        # SMILES round-trip and makes RDKit report the wrong hydrogen
        # count. That corrupted every downstream electronic descriptor
        # (GetTotalNumHs, Gasteiger charges, EState) - the cause of
        # aniline predicting 9.88 instead of its 4.60 training label.
        # Clear it and let RDKit re-derive implicit H from valence.
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    rw.GetAtomWithIdx(idx).SetAtomMapNum(RENDER_TAG)
    try:
        m2 = rw.GetMol()
        m2.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m2)
        smi_tagged = Chem.MolToSmiles(m2)
        rt = Chem.MolFromSmiles(smi_tagged)
        if rt is None: return None, None
    except Exception:
        return None, None
    new_idx = None
    for a in rt.GetAtoms():
        if a.GetAtomMapNum() == RENDER_TAG:
            new_idx = a.GetIdx()
    if new_idx is None: return None, None
    # CRITICAL: strip the tag by TEXT SUBSTITUTION, never by clearing the
    # map and re-canonicalizing. MolToSmiles reorders atoms, so an index
    # found while the tag was present is INVALID for a freshly
    # canonicalized string. That is why predictor.py has _strip_map_tag.
    # This bug made render_site return a carbon index instead of the
    # nitrogen, giving aniline pKa 9.88 instead of its 4.60 label.
    clean = smi_tagged.replace(f":{RENDER_TAG}]", "]")
    return clean, new_idx


ap = argparse.ArgumentParser()
ap.add_argument("smiles")
ap.add_argument("--ph", type=float, default=7.4)
ap.add_argument("--model", default="models/model_core_v16_elec.pkl")
ap.add_argument("--compare", action="store_true",
                help="also show the old independent-site prediction")
args = ap.parse_args()

b = joblib.load(args.model)
gbm, ridge, sc, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")


def predict_pka(prot_smi, prot_idx, dep_smi, dep_idx, kind):
    hg_p, hl_p = p.state_features_v4(prot_smi, prot_idx, kind, n_confs_base=1)
    hg_d, hl_d = p.state_features_v4(dep_smi, dep_idx, kind, n_confs_base=1)
    g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
    l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
    dp = elec_desc(prot_smi, prot_idx); dd = elec_desc(dep_smi, dep_idx)
    if dp is None or dd is None: return None
    feat = np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)
    return electronic.score_hybrid(b, feat)


mol = Chem.MolFromSmiles(args.smiles)
if mol is None: sys.exit(f"could not parse: {args.smiles}")
nm = neutralize(mol)
sites = all_sites(nm)
if not sites: sys.exit("no ionizable sites found")

print(f"\nMolecule : {args.smiles}")
print(f"Sites    : {len(sites)}")
for i, (n, k, idx) in enumerate(sites):
    print(f"  [{TAG0+i}] {n:22s} atom {idx:3d}  {k}")

# tag every site, then build the FULLY PROTONATED starting state
rw = Chem.RWMol(nm)
for i, (n, k, idx) in enumerate(sites):
    rw.GetAtomWithIdx(idx).SetAtomMapNum(TAG0 + i)
work = Chem.MolToSmiles(rw.GetMol())
site_meta = {TAG0 + i: (n, k) for i, (n, k, _) in enumerate(sites)}

n_base = 0
for tag, (name, kind) in site_meta.items():
    if kind == "base":
        nxt = shift_h(work, tag, +1, +1)
        if nxt is not None:
            work = nxt; n_base += 1
        else:
            print(f"  warning: could not protonate base site {tag}")

start_mol = Chem.MolFromSmiles(work)
print(f"\nfully protonated start: charge {Chem.GetFormalCharge(start_mol):+d}")

# --- walk the ladder ---
print(f"\n--- SEQUENTIAL LADDER (each rung on the real charge state) ---")
remaining = set(site_meta.keys())
ladder, warn_charge = [], []
step = 0
while remaining:
    step += 1
    cur_mol = Chem.MolFromSmiles(work)
    cur_q = Chem.GetFormalCharge(cur_mol)
    cands = []
    for tag in sorted(remaining):
        name, kind = site_meta[tag]
        prot_smi, prot_idx = render_site(work, tag)
        dep_work = shift_h(work, tag, -1, -1)
        if prot_smi is None or dep_work is None: continue
        dep_smi, dep_idx = render_site(dep_work, tag)
        if dep_smi is None: continue
        try:
            pk = predict_pka(prot_smi, prot_idx, dep_smi, dep_idx, kind)
        except Exception:
            pk = None
        if pk is not None:
            cands.append((pk, tag, name, kind, dep_work))
    if not cands:
        print(f"  step {step}: no further deprotonation possible"); break
    cands.sort()
    pk, tag, name, kind, dep_work = cands[0]
    others = "  ".join(f"{site_meta[t][0][:12]}={v:.2f}" for v, t, _, _, _ in cands[1:])
    flag = ""
    if abs(cur_q) >= 2:
        flag = "  [!] multiply-charged background - extrapolation"
        warn_charge.append(step)
    print(f"  pKa{step} = {pk:6.2f}   {name:20s} (bg charge {cur_q:+d}){flag}")
    if others:
        print(f"           other candidates: {others}")
    ladder.append({"pKa": pk, "site": name, "kind": kind, "tag": tag})
    work = dep_work
    remaining.discard(tag)

pkas = [r["pKa"] for r in ladder]
mono = sorted(pkas)
if pkas != mono:
    print("\n  note: predicted rungs are not monotonically increasing, which")
    print("  is thermodynamically impossible for a true macro-pKa ladder.")
    print("  Sorting for the distribution below; treat as low confidence.")
    pkas = mono


def distribution(pkas, ph):
    logs, run = [0.0], 0.0
    for pk in pkas:
        run += (ph - pk); logs.append(run)
    mx = max(logs)
    w = [10.0 ** (l - mx) for l in logs]
    t = sum(w)
    return [x / t for x in w]


print(f"\n--- SPECIES DISTRIBUTION AT pH {args.ph} ---")
frac = distribution(pkas, args.ph)
for j, fr in enumerate(frac):
    chg = n_base - j
    lbl = f"charge {chg:+d}" if chg else "neutral"
    print(f"  {lbl:12s} {fr*100:6.2f}%  {'#' * int(round(fr*40))}")
neutral_j = n_base
f_charged = 1.0 - frac[neutral_j] if 0 <= neutral_j < len(frac) else 1.0
print(f"\n  fraction CHARGED at pH {args.ph}: {f_charged*100:.1f}%")

print(f"\n--- pH PROFILE ---")
hdr = "  ".join(f"q{n_base-j:+d}".rjust(6) for j in range(len(pkas)+1))
print(f"{'pH':>5s}  {hdr}")
for ph in [1,2,3,4,5,6,7,7.4,8,9,10,11,12,13]:
    fr = distribution(pkas, ph)
    print(f"{ph:5.1f}  " + "  ".join(f"{x*100:5.1f}%" for x in fr))

if warn_charge:
    print(f"\n[!] rungs {warn_charge} were predicted on backgrounds with |charge| >= 2,")
    print("    outside the mono-ionizable training distribution. Lower confidence.")

if args.compare:
    print(f"\n--- COMPARISON: independent-site (old) vs ladder (new) ---")
    print(f"{'site':22s} {'independent':>12s} {'ladder':>10s}")
    for i, (name, kind, idx) in enumerate(sites):
        tag = TAG0 + i
        prot_smi, prot_idx = render_site(Chem.MolToSmiles(rw.GetMol()), tag)
        base_work = Chem.MolToSmiles(rw.GetMol())
        if kind == "base":
            b2 = shift_h(base_work, tag, +1, +1)
            if b2: prot_smi, prot_idx = render_site(b2, tag)
            dep_smi, dep_idx = render_site(base_work, tag)
        else:
            d2 = shift_h(base_work, tag, -1, -1)
            dep_smi, dep_idx = render_site(d2, tag) if d2 else (None, None)
        indep = None
        if prot_smi and dep_smi:
            try: indep = predict_pka(prot_smi, prot_idx, dep_smi, dep_idx, kind)
            except Exception: pass
        lad = next((r["pKa"] for r in ladder if r["tag"] == tag), None)
        print(f"{name:22s} {('%.2f'%indep) if indep else '   --':>12s} "
              f"{('%.2f'%lad) if lad else '  --':>10s}")

print("""
METHOD NOTE: each rung is predicted on the charge state that actually
exists at that point, so site-site electrostatic coupling enters through
the features. This is the sequential macro-pKa ladder. Uni-pKa instead
predicts a free energy for every microstate and combines them with a
partition function, which additionally handles near-degenerate orderings
and gives true micro-pKa values.""")
