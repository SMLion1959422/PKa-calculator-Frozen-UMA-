"""STEP 2: train a differentiable head with cycle-consistency loss.

WHY A TORCH MLP INSTEAD OF LIGHTGBM
LightGBM cannot express a loss that couples FOUR predictions of the same
molecule. Cycle consistency is exactly such a loss: for a 2-site molecule
the four microstates form a thermodynamic square, and the two paths from
fully-protonated to fully-deprotonated must sum identically:

    pKa(11->01) + pKa(01->00)  ==  pKa(11->10) + pKa(10->00)

That constraint holds regardless of whether we know any experimental
value, so it is SELF-SUPERVISED. Combined with one real label per
molecule it propagates supervision onto the unlabelled transitions -
which are precisely the charged-background cases where training coverage
is 0.27% and the polyprotic benchmark fails.

LOSS = L1(labelled transitions) + lambda * MSE(cycle residual)

HONEST WARNING
Consistency is necessary but NOT sufficient - a model can be perfectly
self-consistent and perfectly wrong. We measured correlation between
cycle residual and actual error at r=0.09 (n=30): none. So the mechanism
that could help here is LABEL PROPAGATION, not consistency-as-target. If
the polyprotic benchmark does not improve on MAE 2.006, this approach
should be discarded rather than tuned. The script trains a mono baseline
alongside so you can tell whether any change came from the cycle loss or
just from swapping LightGBM for an MLP.
"""
import sys
if "venv311" not in sys.prefix:
    sys.exit("WRONG PYTHON: " + sys.prefix + "\n  activate venv311 first")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LAMBDA = 1.0          # weight on cycle-consistency loss; 0.0 = ablation
EPOCHS = 200
LR = 1e-3
HIDDEN = 512
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("loading two-site microstate features...")
data = joblib.load("feat_twosite.pkl")
mols = [k for k, v in data.items() if any(t[3] for t in v["trans"])]
print(f"  {len(data)} molecules, {len(mols)} with a labelled transition")

print("loading mono-ionizable training set (for the supervised anchor)...")
f6 = joblib.load("feat_train_v6.pkl")
elec = joblib.load("feat_electronic.pkl")
corrected = joblib.load("feat_marvin_corrected.pkl")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

def priority_atom(mol):
    for n_, sm, ai in ACID_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    for n_, sm, ai in BASE_SITES:
        pt = Chem.MolFromSmarts(sm)
        if pt is not None:
            m = mol.GetSubstructMatches(pt)
            if m: return m[0][ai]
    return None

valid6 = {s for s, v in f6.items() if np.asarray(v).shape == (2304,)}
mono_X, mono_y = [], []
for mol in Chem.ForwardSDMolSupplier(
        "mlpka/datasets/combined_training_datasets_unique.sdf"):
    if mol is None or not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")):
        continue
    try:
        exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
        smi = Chem.MolToSmiles(mol); nm = neutralize(Chem.Mol(mol))
    except Exception:
        continue
    if not (0 < exp < 14) or ma >= nm.GetNumAtoms() or smi not in elec:
        continue
    pidx = priority_atom(nm)
    if pidx is not None and pidx == ma and smi in valid6:
        mono_X.append(np.concatenate([f6[smi], elec[smi]])); mono_y.append(exp)
    elif smi in corrected:
        mono_X.append(np.concatenate([corrected[smi]["feat"], elec[smi]]))
        mono_y.append(corrected[smi]["pKa"])
mono_X = np.nan_to_num(np.vstack(mono_X)).astype(np.float32)
mono_y = np.array(mono_y, dtype=np.float32)
print(f"  {len(mono_y)} mono-ionizable labelled examples")

scaler = StandardScaler().fit(mono_X)
DIM = mono_X.shape[1]
print(f"  feature dim: {DIM}")

class Head(nn.Module):
    def __init__(self, d, h=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h // 2, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

# split molecules, not transitions, so a molecule never straddles the split
tr_mols, va_mols = train_test_split(mols, test_size=0.15, random_state=SEED)
print(f"  cycle molecules: {len(tr_mols)} train / {len(va_mols)} val")

def pack(mol_list):
    """Return per-molecule tensors: all 4 transition features, the index
    of the labelled one, the label, and the cycle sign vector."""
    out = []
    for k in mol_list:
        v = data[k]
        keys = sorted(v["feats"].keys())
        if len(keys) != 4:
            continue
        F = np.stack([v["feats"][kk] for kk in keys])
        F = scaler.transform(np.nan_to_num(F)).astype(np.float32)
        lab_i = None
        for j, kk in enumerate(keys):
            st, s2 = kk
            for t in v["trans"]:
                if t[0] == st and t[1] == s2 and t[3]:
                    lab_i = j
        if lab_i is None:
            continue
        # cycle: sum over transitions removing site0 then site1, minus
        # the reverse order. Build sign vector from the state pairs.
        sign = np.zeros(4, dtype=np.float32)
        for j, (st, s2) in enumerate(keys):
            i_site = 0 if st[0] != s2[0] else 1
            # path A removes site0 first: (1,1)->(0,1)->(0,0)
            if (st, s2) == ((1,1),(0,1)) or (st, s2) == ((0,1),(0,0)):
                sign[j] += 1.0
            if (st, s2) == ((1,1),(1,0)) or (st, s2) == ((1,0),(0,0)):
                sign[j] -= 1.0
        out.append((torch.tensor(F), lab_i, float(v["label"]), torch.tensor(sign)))
    return out

train_pack = pack(tr_mols)
val_pack = pack(va_mols)
print(f"  usable after packing: {len(train_pack)} / {len(val_pack)}")

mono_Xs = torch.tensor(scaler.transform(mono_X), dtype=torch.float32)
mono_yt = torch.tensor(mono_y)

def run(lam, tag):
    torch.manual_seed(SEED)
    model = Head(DIM)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    l1 = nn.L1Loss()
    best, best_state, patience = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
        # mono supervised batch
        idx = torch.randperm(len(mono_yt))[:512]
        opt.zero_grad()
        loss = l1(model(mono_Xs[idx]), mono_yt[idx])
        # cycle molecules
        cyc_loss = torch.tensor(0.0)
        sup_loss = torch.tensor(0.0)
        if train_pack:
            sub = np.random.choice(len(train_pack), min(64, len(train_pack)), replace=False)
            for j in sub:
                F, li, lab, sign = train_pack[j]
                pred = model(F)
                sup_loss = sup_loss + torch.abs(pred[li] - lab)
                cyc_loss = cyc_loss + (pred * sign).sum() ** 2
            sup_loss = sup_loss / len(sub)
            cyc_loss = cyc_loss / len(sub)
        total = loss + sup_loss + lam * cyc_loss
        total.backward()
        opt.step()

        if ep % 10 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                errs, cycs = [], []
                for F, li, lab, sign in val_pack:
                    pr = model(F)
                    errs.append(abs(pr[li].item() - lab))
                    cycs.append(abs((pr * sign).sum().item()))
                vm = float(np.mean(errs)) if errs else 9e9
                vc = float(np.mean(cycs)) if cycs else 0.0
            if vm < best:
                best, best_state, patience = vm, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
            if ep % 40 == 0:
                print(f"    ep{ep:3d}  val_MAE={vm:.3f}  val_cycle={vc:.3f}")
            if patience >= 6:
                break
    print(f"  {tag}: best val MAE on labelled transition = {best:.3f}")
    return best_state, best

print("\n--- ABLATION: lambda=0 (no cycle loss) ---")
st0, b0 = run(0.0, "lambda=0")
print(f"\n--- WITH cycle loss (lambda={LAMBDA}) ---")
st1, b1 = run(LAMBDA, f"lambda={LAMBDA}")

print(f"\ncycle loss effect on val MAE: {b0:.3f} -> {b1:.3f} ({b1-b0:+.3f})")
best_state = st1 if b1 <= b0 else st0
chosen = LAMBDA if b1 <= b0 else 0.0
model = Head(DIM); model.load_state_dict(best_state); model.eval()
joblib.dump({"state_dict": {k: v.numpy() for k, v in model.state_dict().items()},
             "scaler": scaler, "dim": DIM, "hidden": HIDDEN, "lambda": chosen},
            "models/model_polyprotic_mlp.pkl")
print(f"saved -> models/model_polyprotic_mlp.pkl  (lambda={chosen})")
print("\nNEXT: rerun build_polyprotic_benchmark.py pointed at this model.")
print("If benchmark MAE does not beat 2.006, discard this approach.")
