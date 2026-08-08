"""Generate umapka_finetune.ipynb - end-to-end UMA fine-tuning on Colab.

WHY THIS EXISTS
Every experiment so far optimized AROUND a frozen representation:
    better head (stochastic bagging)  -> 0.949 novartis   (small win)
    better pooling (attention head)   -> 0.627 OOF        (LOST to 0.543)
    better features (xTB descriptors) -> 0.541 vs 0.521   (LOST)
    more data (v21, +6216 molecules)  -> 1.075 novartis   (WORSE)
The attention-head result is the decisive one: letting a model learn
which atoms matter did not help, so the information is not in the frozen
embeddings. The representation itself is the ceiling. This notebook
changes the representation.

WHAT IT DOES DIFFERENTLY
The inference hook detaches (correct for inference, fatal for training),
and even without the detach, get_potential_energy() computes forces whose
internal backward FREES the graph. So this calls UMA's backbone directly:
no forces, graph intact, 138/149 params get gradient. Charge must be
passed via r_data_keys=["charge","spin"] or from_ase silently drops it.

    python make_finetune_notebook.py  ->  umapka_finetune.ipynb
"""
import json

md = lambda s: {"cell_type": "markdown", "metadata": {},
                "source": s.strip().splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip().splitlines(keepends=True)}

cells = [
md("""
# Fine-tuning UMA for pKa

### Why this and not more head-tuning

| approach | result |
|---|---|
| better head (stochastic bagging) | **0.949** novartis - small win |
| better pooling (attention head) | 0.627 OOF - *lost* to 0.543 |
| better features (xTB) | 0.541 vs 0.521 - *lost* |
| more data (+6216 molecules) | 1.075 novartis - *worse* |
| **changing the representation** | **this notebook** |

The attention-head result settled it: letting a model learn *which atoms
matter* did not help, so that information is not present in the frozen
embeddings. Three separate neural approaches have now lost to
gradient-boosted trees on these features. The frozen representation is
the ceiling, so the only remaining move is to change it.

### How this gets gradients into UMA

The inference path cannot be reused. Two blockers, both hit for real:

1. `predictor.py`'s hook calls `.detach()` - severs the graph.
2. Removing the detach is *still* not enough: `get_potential_energy()`
   makes fairchem compute forces, which runs its own `backward` and frees
   the graph. The loss then dies with *"Trying to backward through the
   graph a second time"*.

So we bypass the ASE calculator and energy head and call UMA's
**backbone** directly - no forces, nothing freed, and faster. Verified:
**138 of 149** UMA parameters receive nonzero gradient this way.

### Honest expectations

- 5,184 molecules is a **small** target for fine-tuning a foundation
  model. Overfitting is the main risk, which is why most of UMA stays
  frozen and only the last blocks train.
- The ASE calculator path runs **one structure at a time**, and each
  example needs two (protonated + deprotonated). Expect roughly
  **30-90 min per epoch** on a T4. Plan for 3-6 epochs, not 50.
- This is exploratory. It may not beat 0.949. Cell 5 is a hard gate that
  tells you within a minute whether gradients reach UMA at all - if they
  do not, stop rather than train a head with a frozen trunk by accident.

### Prerequisites
Finish the setup notebook first (`umapka_gpu.ipynb`) so the environment,
Drive mount, and `feat_atomwise.pkl` all exist.
"""),

md("""
## 1. Install

**Every Colab notebook gets its own VM**, so the setup notebook's
installs do NOT carry over here - this notebook has to install
everything itself.

Exact pins, because "latest" broke this three separate ways:

| package | pin | why |
|---|---|---|
| `torch` | `==2.8.0` | fairchem-core 2.21.0 requires `~=2.8.0`. Latest (2.11/2.13) gives `libtorch_cuda_linalg.so: undefined symbol` from `torch.det` deep inside UMA's forward pass. |
| `setuptools` | `<81` | setuptools **removed `pkg_resources`** after v81; something in the import chain still needs it (`pkgutil.ImpImporter` error on Python 3.12). |
| `numpy` | `<2.5` | fairchem-core requires `<2.5,>=2.0`. |

Run this cell, then **Runtime > Restart session**, then continue at
cell 2. Red dependency-conflict warnings from unrelated Colab packages
(cudf, pytensor, datasets) are expected and harmless.
"""),
code("""
!pip install -q "fairchem-core==2.21.0" "rdkit==2026.3.4" "lightgbm==4.7.0" \\
                "scikit-learn==1.9.0" "ase==3.29.0" joblib tqdm
!pip install --force-reinstall --no-cache-dir "torch==2.8.0" "setuptools<81" \\
             "numpy<2.5,>=2.0" packaging
print("\\nDONE - now Runtime > Restart session, then continue at cell 2")
"""),

md("## 2. Verify the environment before spending GPU time"),
code("""
import torch, numpy, setuptools, pkg_resources, sys
print("python        :", sys.version.split()[0])
print("torch         :", torch.__version__, "| cuda:", torch.version.cuda,
      "| avail:", torch.cuda.is_available())
print("numpy         :", numpy.__version__, "(must be < 2.5)")
print("setuptools    :", setuptools.__version__, "(must be < 81)")
print("pkg_resources :", pkg_resources.__file__)
print("cuda linalg   :", torch.det(torch.eye(3, device="cuda")).item(), "(must be 1.0)")
assert torch.__version__.startswith("2.8"), "wrong torch - re-run cell 1 then restart"
assert torch.cuda.is_available(), "Runtime > Change runtime type > T4 GPU"
print("GPU           :", torch.cuda.get_device_name(0))
print("free VRAM (GB):", round(torch.cuda.mem_get_info()[0]/1e9, 2))
print("\\nALL CHECKS PASSED")
"""),

code("""
from google.colab import drive
drive.mount('/content/drive')
!cp -n /content/drive/MyDrive/colab_bundle.zip /content/ 2>/dev/null
!unzip -o -q /content/colab_bundle.zip -d /content/pka
%cd /content/pka
!ls umapka models mlpka/datasets | head
"""),

md("## 3. Load UMA and the labelled pairs"),
code("""
from huggingface_hub import login
login()   # token from https://huggingface.co/settings/tokens
"""),

code("""
import numpy as np, torch, joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
RDLogger.DisableLog("rdApp.*")

from umapka import PkaPredictor
from umapka.predictor import (protonation_pair_site_tagged,
                               _smiles_to_atoms_with_site)

p = PkaPredictor("models/model_core_v3.pkl")
print("device:", p.device)
assert p.device == "cuda"

df = PandasTools.LoadSDF("mlpka/datasets/combined_training_datasets_unique.sdf")
pk_col = next(c for c in df.columns if c.lower() in ("pka","pka_value","value"))
pairs = []
for _, r in df.iterrows():
    m = r.get("ROMol")
    if m is None: continue
    try:
        v = float(r[pk_col]); smi = Chem.MolToSmiles(m)
    except Exception: continue
    if 0 < v < 14:
        pairs.append((smi, v))
seen=set(); pairs=[x for x in pairs if not (x[0] in seen or seen.add(x[0]))]
print("labelled molecules:", len(pairs))
"""),

md("""
## 4. Gradient-enabled embedding extraction

**Do NOT reuse `PkaPredictor.embeddings()` or its forward hook here.**
Two problems, both found by actually running them:

1. `predictor.py`'s hook calls `.detach()`, severing the graph.
2. Removing the detach is still not enough: `atoms.get_potential_energy()`
   makes fairchem compute **forces**, which runs its own `backward` and
   **frees the graph**. Your loss then dies with
   *"Trying to backward through the graph a second time"*.

So we skip the ASE calculator and the energy head entirely and call the
**backbone** directly. No force computation, nothing freed, and it is
faster because the output head never runs.

`node_embedding` is `(n_atoms, 9, 128)` - 9 spherical-harmonic channels.
Channel 0 is the scalar/invariant part, which is the per-atom
representation we want.

`r_data_keys=["charge", "spin"]` is **mandatory and easy to miss**.
Without it `from_ase` silently drops charge: the same geometry at
charge 0 and charge -1 produced embeddings differing by 3.6e-07 (pure
numerical noise). With it, 3.9e-01. pKa *is* a charge-changing
transition, so a model blind to charge is worthless - and it would have
trained for hours looking perfectly healthy.
"""),
code("""
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

INNER = p._calc.predictor.tracked_modules()["model"].module

def embed_grad(atoms):
    \"\"\"(n_atoms, 128) per-atom UMA features with the autograd graph intact.
    atoms.info must carry {'charge': int, 'spin': int} - the helpers in
    umapka.predictor already set these correctly per protonation state.\"\"\"
    ad = AtomicData.from_ase(
        atoms, task_name="omol", molecule_cell_size=120.0,
        r_energy=False, r_forces=False, r_stress=False,
        r_data_keys=["charge", "spin"],          # <- without this, charge is dropped
    )
    batch = atomicdata_list_to_batch([ad]).to(p.device)
    return INNER.backbone(batch)["node_embedding"][:, 0, :].float()

def pair_atoms(smi):
    prot, pi_, dep, di_, kind = protonation_pair_site_tagged(smi, return_kind=True)
    ap, sp, _ = _smiles_to_atoms_with_site(prot, pi_)
    ad_, sd, _ = _smiles_to_atoms_with_site(dep, di_)
    return (ap, sp), (ad_, sd), kind

print("ready")
"""),

md("""
## 5. HARD GATE - gradients AND charge

Runs in under a minute and checks the two things that silently ruin this:

1. **Gradients reach UMA.** If not, training optimizes only the head on a
   frozen trunk - a slower rerun of the attention-head experiment that
   already failed at 0.627.
2. **Charge is propagated.** If not, the two protonation states are
   effectively indistinguishable to the model.

**Do not continue past a FAIL.**
"""),
code("""
(ap, sp), (ad_, sd), kind = pair_atoms("CC(=O)O")
print("prot info:", ap.info, "| deprot info:", ad_.info)

h = embed_grad(ap)
print("embedding:", tuple(h.shape), "| requires_grad:", h.requires_grad,
      "| grad_fn:", type(h.grad_fn).__name__ if h.grad_fn is not None else None)

named = list(INNER.named_parameters())
print("total UMA params:", len(named))

for _, q in named: q.grad = None
h.pow(2).mean().backward()
got = [(n, float(q.grad.abs().sum())) for n, q in named
       if q.grad is not None and float(q.grad.abs().sum()) > 0]
print("params with NONZERO grad:", len(got), "of", len(named))
for n, g in got[:4]: print(f"   {n[:66]:66s} {g:.3e}")

# charge sensitivity: same geometry, different charge, must differ
a_neg = ap.copy(); a_neg.info = dict(ap.info); a_neg.info["charge"] = -1
with torch.no_grad():
    d = (embed_grad(ap) - embed_grad(a_neg)).abs().max().item()
print(f"charge 0 vs -1, same geometry -> max diff {d:.3e}")

assert h.grad_fn is not None,  "FAIL: no grad_fn - graph severed"
assert len(got) > 0,           "FAIL: no UMA parameter received gradient"
assert d > 1e-4,               "FAIL: charge not propagated - check r_data_keys"
print("\\nPASS - gradients reach UMA and charge is encoded.")
"""),

md("""
## 6. Choose what to train

Fine-tuning *everything* on 5k molecules would overfit badly and will not
fit a T4's memory. Freeze the trunk, unfreeze only the top blocks.

Start with `N_UNFROZEN = 2`. If loss will not move, raise it; if it
overfits (train loss falls, val rises), lower it.
"""),
code("""
N_UNFROZEN = 2

all_params = list(INNER.named_parameters())
for _, q in all_params: q.requires_grad_(False)

# unfreeze the deepest N blocks by matching the numeric block index
import re
idx = []
for n, _ in all_params:
    m = re.search(r"blocks?\\.(\\d+)\\.", n)
    if m: idx.append(int(m.group(1)))
top = sorted(set(idx))[-N_UNFROZEN:] if idx else []
print("unfreezing block indices:", top)

n_train = 0
for n, q in all_params:
    m = re.search(r"blocks?\\.(\\d+)\\.", n)
    if (m and int(m.group(1)) in top) or "energy" in n.lower():
        q.requires_grad_(True); n_train += q.numel()
print(f"trainable UMA params: {n_train/1e6:.2f} M of "
      f"{sum(q.numel() for _, q in all_params)/1e6:.1f} M")
"""),

md("## 7. The pKa head"),
code("""
import torch.nn as nn

class PkaHead(nn.Module):
    \"\"\"Consumes per-atom embeddings for BOTH protonation states.

    pKa is a property of the transition, not of a molecule - the frozen
    pipeline encodes [h_prot ; h_deprot ; h_prot - h_deprot] for exactly
    this reason, and that convention is kept here. Site-local pooling is
    included because it beat global-only pooling in a controlled A/B
    (0.657 -> 0.591).\"\"\"
    def __init__(self, d=128, hidden=256):
        super().__init__()
        # pool() returns 3*d per state (mean + max + site-local); the
        # head sees [prot ; deprot ; prot-deprot] = 9*d
        self.net = nn.Sequential(
            nn.Linear(d*9, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
    @staticmethod
    def pool(h, site):
        g = torch.cat([h.mean(0), h.max(0).values])          # global: 2*d
        loc = h[site]                                          # titratable atom: d
        return torch.cat([g, loc])                             # 3*d
    def forward(self, hp, sp, hd, sd):
        a = self.pool(hp, sp); b = self.pool(hd, sd)
        return self.net(torch.cat([a, b, a - b]).unsqueeze(0)).squeeze()

head = PkaHead().to(p.device)
print("head params:", sum(q.numel() for q in head.parameters()))

# shape check on synthetic data - cheaper than finding out mid-epoch
_hp, _hd = torch.randn(11, 128, device=p.device), torch.randn(10, 128, device=p.device)
print("head output:", head(_hp, 3, _hd, 3).shape, "(must be scalar, torch.Size([]))")
"""),

md("""
## 8. Train

Two forward passes per molecule, one structure at a time (the ASE path
does not batch), so expect **30-90 min/epoch**. `SUBSET` keeps the first
run short - confirm the loss actually moves before committing hours.

Two learning rates: UMA needs a much smaller one than a randomly
initialised head, or the pretrained weights get destroyed in the first
few steps.
"""),
code("""
import random, time, math

SUBSET   = 1500     # raise to len(pairs) once you see the loss move
EPOCHS   = 3
ACCUM    = 4        # gradient accumulation (effective batch size)
LR_UMA   = 1e-5
LR_HEAD  = 1e-3
VAL_FRAC = 0.15

random.seed(42)
data = pairs[:SUBSET][:]
random.shuffle(data)
n_val = int(len(data)*VAL_FRAC)
val, train = data[:n_val], data[n_val:]
print(f"train {len(train)} | val {len(val)}")

opt = torch.optim.AdamW([
    {"params": [q for _, q in all_params if q.requires_grad], "lr": LR_UMA},
    {"params": head.parameters(), "lr": LR_HEAD},
], weight_decay=1e-4)

def forward_one(smi):
    (ap, sp), (ad_, sd), _k = pair_atoms(smi)
    hp = embed_grad(ap)
    hd = embed_grad(ad_)
    return head(hp, sp, hd, sd)

def evaluate(rows):
    head.eval(); errs = []
    with torch.no_grad():
        for smi, y in rows:
            try: errs.append(abs(float(forward_one(smi)) - y))
            except Exception: pass
    head.train()
    return float(np.mean(errs)) if errs else float("nan")

print("baseline (untrained head) val MAE:", round(evaluate(val[:60]), 3))

for ep in range(EPOCHS):
    t0 = time.time(); run = []; opt.zero_grad()
    for i, (smi, y) in enumerate(train):
        try:
            pred = forward_one(smi)
            loss = nn.functional.smooth_l1_loss(
                pred, torch.tensor(float(y), device=p.device))
            (loss / ACCUM).backward()
            run.append(float(loss))
        except Exception:
            continue
        if (i+1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [q for _, q in all_params if q.requires_grad] +
                list(head.parameters()), 1.0)
            opt.step(); opt.zero_grad()
        if (i+1) % 200 == 0:
            print(f"  ep{ep+1} [{i+1}/{len(train)}] "
                  f"loss {np.mean(run[-200:]):.3f}  {(time.time()-t0)/60:.1f} min",
                  flush=True)
    vm = evaluate(val)
    print(f"epoch {ep+1}: train loss {np.mean(run):.3f} | val MAE {vm:.3f} "
          f"| {(time.time()-t0)/60:.1f} min")
    torch.save({"head": head.state_dict(),
                "uma": {n: q.detach().cpu() for n, q in all_params if q.requires_grad}},
               f"/content/drive/MyDrive/finetune_ep{ep+1}.pt")
    print("  checkpoint -> Drive")
"""),

md("""
## 9. Did it actually beat the frozen model?

The frozen baseline on this *same* held-out slice is the only fair
comparison - the published 0.949 is on Novartis, a different set.

If val MAE is not clearly below the frozen baseline, fine-tuning is not
working: try `N_UNFROZEN = 4`, or `LR_UMA = 3e-5`, or more epochs. If it
still will not move, the honest conclusion is that 5k molecules is too
small to fine-tune this model, and **v20 at 0.949 stands**.
"""),
code("""
import joblib
from umapka import electronic

b = joblib.load("models/model_core_v20_ensemble.pkl")
errs = []
for smi, y in val:
    try:
        prot, pi_, dep, di_, kind = protonation_pair_site_tagged(smi, return_kind=True)
        feat = electronic.build_hybrid_features(p, prot, pi_, dep, di_, kind)
        if feat is None: continue
        errs.append(abs(electronic.score_any(b, feat) - y))
    except Exception:
        pass
print(f"FROZEN v20 on this val slice : {np.mean(errs):.3f}  (n={len(errs)})")
print(f"FINE-TUNED on this val slice : {evaluate(val):.3f}")
print("\\nNote: these are TRAINING-DISTRIBUTION molecules, so both numbers")
print("flatter. The real test is Novartis - run cell 9 before believing it.")
"""),

md("""
## 10. The real test - Novartis

**Do not skip this.** Four separate times, gains measured on
training-distribution data produced zero or negative movement on
Novartis, because training data and AvLiLuMoVe are near-neighbours
(median Tanimoto 0.708) while Novartis is genuinely novel (0.330).

Score it **once**, at the end. Do not tune against it - on 275 molecules
that is fitting noise, and it is the trap that had the original
`model_core.pkl` reporting 0.561 when the truth was 1.41.

Beat **0.949** and the fine-tuning was real.
"""),
code("""
from rdkit import Chem
errs_ft = []
mols = [m for m in Chem.ForwardSDMolSupplier(
    "mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf") if m is not None]
head.eval()
with torch.no_grad():
    for m in mols:
        if not m.HasProp("pKa"): continue
        try:
            y = float(m.GetProp("pKa"))
            if not (0 < y < 14): continue
            errs_ft.append(abs(float(forward_one(Chem.MolToSmiles(m))) - y))
        except Exception:
            pass
print(f"FINE-TUNED novartis MAE : {np.mean(errs_ft):.3f}  (n={len(errs_ft)})")
print( "frozen v20 novartis MAE : 0.949")
print( "ChemAxon Marvin         : 0.856  (uses its own site annotations)")
print( "target                  : 0.800")
"""),
]

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "cells": cells}

with open("umapka_finetune.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote umapka_finetune.ipynb ({len(cells)} cells)")
