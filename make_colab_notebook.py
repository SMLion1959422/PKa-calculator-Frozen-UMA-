"""Generate umapka_gpu.ipynb - a real Colab notebook for the GPU work.

COLAB.md is the prose version; this emits the same pipeline as an actual
notebook so it can be opened with File > Upload notebook and run
top-to-bottom, instead of copy-pasting cells by hand.

    python make_colab_notebook.py   ->  umapka_gpu.ipynb
"""
import json

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip().splitlines(keepends=True)}

cells = [
md("""
# umapka on GPU

Every expensive step here is UMA inference. On CPU, caching per-atom
embeddings ran at **250 molecules / 12 min (~4.5 h total)**. On a Colab
T4 expect **15-40 min**.

`PkaPredictor` already does `device = "cuda" if torch.cuda.is_available()
else "cpu"`, so **no code changes are needed** - the same scripts just
run faster.

### Baselines to beat (held-out, learned site finder, no ChemAxon)

| config | Novartis | AvLiLuMoVe |
|---|---|---|
| **v20 ensemble (current best)** | **0.949** | 0.411 |
| v16 hybrid | 0.965 | 0.449 |
| v21 + 6216 extra molecules | 1.075 | 0.284 |
| ChemAxon Marvin *(uses its own site annotations)* | 0.856 | 0.566 |
| oracle-site ceiling, frozen features | 0.845 | - |
| **wall: site AND kind both correct** | **0.842** | - |

**Novartis is the number that matters** - it is genuine extrapolation
(median nearest-neighbour Tanimoto to training 0.330). AvLiLuMoVe is
largely interpolation (0.708, 28% near-duplicates), so a model can look
great there while getting worse where it counts. v21 is exactly that
trap: adding clean data improved AvLiLuMoVe 31% and made Novartis worse.

**Steps 1-3 below are worth a few percent. Only Step 4 (fine-tuning UMA)
can plausibly reach 0.80**, because the 0.842 wall *is* the frozen
representation.

---
### Before you start
1. **Runtime > Change runtime type > GPU** (T4 ok, L4/A100 better)
2. Upload `colab_bundle.zip` via the folder icon in the left sidebar
3. Accept the license at https://huggingface.co/facebook/UMA (the model
   is gated; the download 401s otherwise)
"""),

md("## 1. Confirm you actually got a GPU"),
code("""
import subprocess, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU"
print("device:", torch.cuda.get_device_name(0))
"""),

md("## 2. Unpack the bundle"),
code("""
!unzip -o -q /content/colab_bundle.zip -d /content/pka
%cd /content/pka
!ls umapka models mlpka/datasets
"""),

md("""
## 3. Install dependencies

Versions are pinned deliberately. Two mismatches cost real debugging time
on CPU: a lightgbm 3.x pickle silently degraded the site finder to a much
weaker fallback, and scikit-learn 1.3.2 (vs the pinned 1.9.0) made
LightGBM `.fit()` die with a memory access violation.

**Installing `fairchem-core` lets pip swap torch for a build whose CUDA
sub-libraries do not match Colab's**, which shows up later as:

```
RuntimeError: Error in dlopen: .../libtorch_cuda_linalg.so:
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

...thrown from `torch.det()` deep inside UMA's forward pass, *after* the
1.17 GB checkpoint download. So torch is force-reinstalled as a
consistent set immediately afterwards, before anything imports it.

**After this cell: Runtime > Restart session, then re-run cell 2.**
"""),
code("""
import sys, torch
print("python:", sys.version)
print("torch before:", torch.__version__, "| cuda:", torch.version.cuda)

!pip -q install "fairchem-core==2.21.0" "rdkit==2026.3.4" \\
                "lightgbm==4.7.0" "scikit-learn==1.9.0" \\
                "ase==3.29.0" joblib tqdm

# Repair torch: reinstall the three packages together so their CUDA
# libraries come from one consistent build. Without this, torch.det()
# inside UMA dies with an undefined-symbol dlopen error.
!pip -q install --force-reinstall --no-cache-dir torch torchvision torchaudio

# Colab is on Python 3.12, which REMOVED pkgutil.ImpImporter. Ubuntu's
# system setuptools at /usr/lib/python3/dist-packages still references
# it, and the reinstall above can shift import precedence onto that old
# copy - which then fails with
#   AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
# from inside pkg_resources, 14 frames deep in the fairchem import.
# A pip-installed modern setuptools shadows the system one.
!pip -q install --upgrade --force-reinstall setuptools packaging

print("\\nNow: Runtime > Restart session, then re-run cell 2 before continuing.")
"""),

md("""
### 3b. Verify the environment BEFORE loading UMA

Run this straight after the restart. It checks the two things that break
a fresh Colab install, in one second each - as opposed to discovering
them 14 frames deep after a 1.17 GB checkpoint download.

1. `torch.det` on a CUDA tensor - the exact op that dies when torch's
   CUDA sub-libraries are mismatched.
2. Which `pkg_resources` wins - the system copy at
   `/usr/lib/python3/dist-packages` is pre-3.12 and raises
   `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`.
"""),
code("""
import torch, setuptools, pkg_resources
print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
      "| available:", torch.cuda.is_available())
d = torch.det(torch.eye(3, device="cuda"))
print("det on cuda:", d.item(), "(must be 1.0)")
assert abs(d.item() - 1.0) < 1e-6, "CUDA linalg broken - re-run the torch reinstall"

print("setuptools:", setuptools.__version__)
print("pkg_resources:", pkg_resources.__file__)
assert "/usr/lib/python3/" not in pkg_resources.__file__, \\
    "system (pre-3.12) pkg_resources is shadowing pip's - re-run the setuptools upgrade"
print("\\nenvironment OK")
"""),

md("## 4. HuggingFace auth (required - `uma-s-1p1` is gated)"),
code("""
from huggingface_hub import login
login()   # token from https://huggingface.co/settings/tokens
"""),

md("""
## 5. Smoke test - do this BEFORE spending GPU time

If `device` prints `cpu`, stop and fix the runtime. Otherwise you are
about to burn hours at CPU speed.
"""),
code("""
from umapka import PkaPredictor
p = PkaPredictor("models/model_core_v20_ensemble.pkl")
print("device:", p.device)
assert p.device == "cuda", "PkaPredictor is on CPU - fix the runtime first"
for smi, ref, name in [("CC(=O)O", 4.76, "acetic acid"),
                       ("Oc1ccccc1", 9.99, "phenol"),
                       ("c1ccncc1", 5.23, "pyridine")]:
    print(f"{name:14s} {p.predict(smi):5.2f}   (ref {ref})")
"""),

md("""
## 6. Step 1 - per-atom embedding cache

Prerequisite for steps 2 and 4. **Resumable** (checkpoints every 250), so
a disconnect costs at most 250 molecules - just re-run this cell.

Watch the printed rate: if it is not far better than 250 molecules /
12 min, the GPU is not being used.
"""),
code("""
!python -u cache_atom_embeddings.py
"""),

md("""
## 7. Step 2 - attention-pooling head

Replaces fixed mean/max pooling with pooling the model *learns*, which
targets the `>30`-atom bucket (MAE 1.065) where error concentrates.

Prints 5-fold OOF MAE on the **same split** the tree models used, so it
is directly comparable:

```
LightGBM on pooled features : 0.577
3-model average ensemble    : 0.543
```

If this is not clearly below **0.543**, fixed pooling was not the binding
constraint and the frozen-UMA ceiling is the real one. That is a genuine
result, not a failure - record it and go to Step 4.
"""),
code("""
!python -u train_attention_head.py
"""),

md("""
## 8. Step 3 - multi-conformer averaging

Everything so far used one conformer. This averages the pooled embedding
over 3 conformers for base sites, attacking **geometry noise** rather
than model fit - which matters, because every fit-improving change so far
failed to transfer to Novartis.

Caveat: the models were *trained* on 1-conformer features, so a
train/test mismatch may eat the gain. If 3-conformer test features help,
rebuild the training features the same way and retrain.
"""),
code("""
src = open("cache_external_features.py").read()
src = src.replace("p, prot, pi_, dep, di_, kind)",
                  "p, prot, pi_, dep, di_, kind, n_confs_base=3)")
src = src.replace("feat_external_learned.pkl", "feat_external_3conf.pkl")
open("cache_external_features_3conf.py", "w").write(src)
print("wrote cache_external_features_3conf.py")
"""),
code("""
!python -u cache_external_features_3conf.py
"""),
code("""
import joblib, numpy as np
from umapka import electronic
ext = joblib.load("feat_external_3conf.pkl")
print("3-conformer features   (1-conf baseline: novartis 0.949 | avlilumove 0.411)\\n")
for name in ["models/model_core_v20_ensemble.pkl", "models/model_core_v16_elec.pkl"]:
    b = joblib.load(name)
    print(name.split("/")[-1])
    for ds, rows in ext.items():
        X = np.vstack([np.asarray(r["feat"], dtype=np.float64).reshape(1, -1) for r in rows])
        y = np.array([r["exp"] for r in rows], dtype=float)
        pred = electronic.score_any_batch(b, X)
        print(f"   {ds:12s} n={len(y):4d}  MAE {np.abs(pred - y).mean():.3f}")
"""),

md("""
## 9. Step 4 - fine-tuning UMA (the actual lever)

This is a project, not a cell. Treat the below as a starting point.

**Why.** Today UMA is frozen and only a tree head trains; the 0.842 wall
*is* the frozen representation. Fine-tuning lets the representation adapt
to pKa. Uni-pKa/Starling get their accuracy exactly this way.

**Where to hook in.** `umapka/predictor.py` already grabs the tensor
feeding UMA's energy head:

```python
module = predictor.tracked_modules()["model"]
self._energy_head = module.module.output_heads["energyandforcehead"].head.energy_block
```

`embeddings()` registers a forward hook there. For fine-tuning you need
those vectors **with gradients**:

1. Do **not** wrap in `torch.no_grad()` (the current path does).
2. Put a small head on the pooled embedding - start from
   `train_attention_head.py`'s `PkaNet`, which already consumes per-atom
   vectors + site distance.
3. Backprop into UMA with a much lower LR than the head (~`1e-5` vs
   `1e-3`), and freeze the lower blocks for the first epoch or two.
4. Feed the paired protonated/deprotonated states, as the frozen pipeline
   does - pKa is a property of the *transition*, not a molecule.

**Cautions**
- Memory: `uma-s-1p1` + activations for two states per example will not
  fit a large batch on a T4. Batch 2-4 with gradient accumulation, or A100.
- Overfitting: 5,184 molecules against a foundation model is a tiny
  target. Freeze most of UMA; train the last block(s) first.
- **Keep the honest protocol.** Select on OOF, score Novartis once. Four
  separate times, gains on cross-validated training fit gave zero or
  negative Novartis movement. Do not tune against Novartis - on 275
  molecules that is fitting noise, and it is the trap that had the
  original `model_core.pkl` reporting 0.561 when the truth was 1.41.
"""),

md("## 10. Save results (Colab wipes local storage on disconnect)"),
code("""
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/umapka_out
!cp -v feat_atomwise.pkl attention_head_oof.pkl feat_external_3conf.pkl \\
      /content/drive/MyDrive/umapka_out/ 2>/dev/null || true
!ls -la /content/drive/MyDrive/umapka_out/
"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

with open("umapka_gpu.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"wrote umapka_gpu.ipynb ({len(cells)} cells)")
