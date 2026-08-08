# Running the GPU work in Google Colab

## Why GPU changes anything

Every expensive step in this project is UMA inference. On this CPU box,
caching per-atom embeddings for 5,994 molecules was running at
**250 molecules / 12 min → ~4.5 hours**. On a Colab T4 that should be
**15–40 minutes**, and on an L4/A100 less.

`PkaPredictor.__init__` already does
`device = "cuda" if torch.cuda.is_available() else "cpu"`, so **no code
changes are needed** — the same scripts just run faster.

What the GPU unlocks, in order of expected value:

| step | what it attacks | why it might beat 0.842 |
|---|---|---|
| 1. per-atom embedding cache | prerequisite | enables steps 2–3 |
| 2. attention-pooling head | `>30`-atom bucket (MAE 1.065) | learns *which* atoms matter instead of fixed mean/max pooling |
| 3. multi-conformer averaging | geometry noise | attacks noise, not fit — the failure mode that plagued the CPU attempts |
| 4. UMA fine-tuning | the frozen-feature ceiling itself | the only lever with real headroom; this is what Uni-pKa/Starling do |

**Honest framing:** steps 2–3 are worth a few percent. The wall is
**0.842** (MAE on molecules whose site *and* kind are already correct),
so only **step 4** can plausibly reach 0.80. Steps 1–3 are the cheap
things to do first, and step 1 is a prerequisite for step 4 anyway.

---

## Step 0 — build and upload the bundle

Locally:

```bash
python make_colab_bundle.py
```

That writes `colab_bundle.zip` (~9 MB). It deliberately excludes the big
caches (`feat_train_v6.pkl` is 127 MB) — those are *outputs* of the GPU
run, faster to regenerate than to upload.

In Colab: **Runtime → Change runtime type → GPU** (T4 is fine, L4/A100
better), then upload the zip via the file browser sidebar.

---

## Cell 1 — confirm you actually got a GPU

```python
import torch, subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU - Runtime > Change runtime type > GPU"
print("device:", torch.cuda.get_device_name(0))
```

## Cell 2 — unpack

```python
!unzip -o -q colab_bundle.zip -d /content/pka
%cd /content/pka
!ls umapka models mlpka/datasets
```

## Cell 3 — install dependencies

Version pinning matters here. Two version mismatches cost real debugging
time on the CPU box: a lightgbm 3.x pickle silently degraded the site
finder to a much weaker fallback, and scikit-learn 1.3.2 (vs the pinned
1.9.0) made LightGBM `.fit()` die with a memory access violation.

```python
# fairchem-core needs Python <= 3.12; Colab is currently 3.11/3.12 (fine)
import sys; print(sys.version)

!pip -q install "fairchem-core==2.21.0" "rdkit==2026.3.4" \
                "lightgbm==4.7.0" "scikit-learn==1.9.0" \
                "ase==3.29.0" joblib tqdm
```

Then **Runtime → Restart session** (numpy/torch get rebound), and re-run
Cell 2 to get back to `/content/pka`.

## Cell 4 — HuggingFace auth (required)

`uma-s-1p1` is a **gated** model. You must accept the license at
<https://huggingface.co/facebook/UMA> with the same account, or the
download 401s.

```python
from huggingface_hub import login
login()          # paste a token from huggingface.co/settings/tokens
```

## Cell 5 — smoke test before spending GPU time

```python
from umapka import PkaPredictor
p = PkaPredictor("models/model_core_v20_ensemble.pkl")
print("device:", p.device)          # must say cuda
print("acetic acid:", round(p.predict("CC(=O)O"), 2), "(ref 4.76)")
print("phenol:     ", round(p.predict("Oc1ccccc1"), 2), "(ref 9.99)")
```

If `p.device` says `cpu`, stop — you are about to burn hours at CPU speed.

## Cell 6 — Step 1: per-atom embedding cache

This is the prerequisite for everything else. It is **resumable**
(checkpoints every 250 molecules), so a disconnect costs at most 250.

```python
!python -u cache_atom_embeddings.py
```

Watch the printed rate. If it is not far better than
`250 molecules / 12 min`, the GPU is not being used.

Colab Pro disconnects on idle — keep the tab active, and just re-run the
cell if it drops.

## Cell 7 — Step 2: attention-pooling head

```python
!python -u train_attention_head.py
```

This prints a 5-fold OOF MAE against the **same split** used by the
tree models, so it is directly comparable:

```
LightGBM on pooled features : 0.577
3-model average ensemble    : 0.543
```

If attention pooling is not clearly below **0.543**, then fixed pooling
was not the binding constraint, and the frozen-UMA ceiling is the real
one. That is a genuine result, not a failure — record it and move to
step 4.

## Cell 8 — Step 3: multi-conformer averaging

`state_features_v4(..., n_confs_base=N)` averages the pooled embedding
over `N` conformers for base sites. Everything so far used `N=1`. This
attacks *geometry noise* rather than model fit, which matters because
every fit-improving change so far failed to transfer to Novartis.

```python
# re-cache the held-out features with 3 conformers instead of 1
import re
src = open("cache_external_features.py").read()
src = src.replace("build_hybrid_features(\n                    p, prot, pi_, dep, di_, kind)",
                  "build_hybrid_features(\n                    p, prot, pi_, dep, di_, kind, n_confs_base=3)")
open("cache_external_features_3conf.py","w").write(
    src.replace('feat_external_learned.pkl', 'feat_external_3conf.pkl'))

!python -u cache_external_features_3conf.py
```

```python
# score the existing models against the 3-conformer features
import joblib, numpy as np
from umapka import electronic
ext = joblib.load("feat_external_3conf.pkl")
for name in ["models/model_core_v20_ensemble.pkl", "models/model_core_v16_elec.pkl"]:
    b = joblib.load(name)
    print(name.split("/")[-1])
    for ds, rows in ext.items():
        X = np.vstack([np.asarray(r["feat"], dtype=np.float64).reshape(1,-1) for r in rows])
        y = np.array([r["exp"] for r in rows])
        pred = electronic.score_any_batch(b, X)
        print(f"   {ds:12s} MAE {np.abs(pred-y).mean():.3f}")
```

Baselines to beat (1 conformer): **novartis 0.949**, avlilumove 0.411.

Note: the models were *trained* on 1-conformer features, so a train/test
mismatch may eat the gain. If 3-conformer test features help, rebuild the
training features the same way and retrain.

---

## Step 4 — fine-tuning UMA (the actual lever)

This is a project, not a cell, so treat the sketch below as a starting
point rather than something to paste and trust.

**The idea.** Today UMA is frozen and only a tree head is trained; the
0.842 wall is the frozen representation. Fine-tuning lets the
representation itself adapt to pKa. Uni-pKa gets its accuracy exactly
this way (pretrain on ~1M predicted pKa, then finetune end-to-end with
thermodynamic consistency enforced).

**Where to hook in.** `umapka/predictor.py` already grabs the tensor
feeding UMA's energy head:

```python
module = predictor.tracked_modules()["model"]
self._energy_head = module.module.output_heads["energyandforcehead"].head.energy_block
```

`embeddings()` registers a forward hook there to read per-atom vectors.
For fine-tuning you instead need those vectors **with gradients**, so:

1. Do **not** wrap in `torch.no_grad()` (the current embedding path does).
2. Put a small head on the pooled embedding (start from
   `train_attention_head.py`'s `PkaNet` — it already takes per-atom
   vectors + site distance).
3. Backprop into UMA with a **much** lower LR than the head
   (e.g. `1e-5` for UMA, `1e-3` for the head) and freeze the lower
   blocks for the first epoch or two.
4. Feed the paired protonated/deprotonated states, as the frozen pipeline
   already does — pKa is a property of the *transition*, not a molecule.

**Practical cautions**
- Memory: `uma-s-1p1` plus activations for two states per example will
  not fit a large batch on a T4. Use batch size 2–4 with gradient
  accumulation, or get an A100.
- Overfitting: 5,184 molecules against a foundation model is a very
  small target. Freeze most of UMA; train only the last block(s) first.
- **Keep the same honest protocol.** Select on OOF; score Novartis once.
  Four separate times, gains on cross-validated training fit produced
  zero or negative Novartis gains, because training data and AvLiLuMoVe
  are near-neighbours (median Tanimoto 0.708) while Novartis is genuinely
  novel (0.330). Do not tune against Novartis — on 275 molecules that is
  fitting noise, and it is the trap that made the original
  `model_core.pkl` report 0.561 when the truth was 1.41.

---

## Getting results back

```python
from google.colab import files
files.download("feat_atomwise.pkl")        # if you want it locally
files.download("attention_head_oof.pkl")
# or persist to Drive:
# from google.colab import drive; drive.mount("/content/drive")
# !cp feat_atomwise.pkl /content/drive/MyDrive/
```

## Current baselines (so you know what beating it looks like)

| config | Novartis | AvLiLuMoVe |
|---|---|---|
| v20 ensemble (**current best, in the CLI**) | **0.949** | 0.411 |
| v16 hybrid | 0.965 | 0.449 |
| v21 + 6216 extra molecules | 1.075 | 0.284 |
| ChemAxon Marvin (*uses its own site annotations*) | 0.856 | 0.566 |
| oracle-site ceiling, frozen features | 0.845 | — |
| **wall: correct site + kind only** | **0.842** | — |

Novartis is the number that matters — it is genuine extrapolation
(median NN Tanimoto to training 0.330). AvLiLuMoVe is largely
interpolation (0.708, with 28% near-duplicates), so a model can look
great there while getting worse where it counts. v21 is exactly that
trap: adding data improved AvLiLuMoVe 31% and made Novartis worse.
