# models/

Trained regressor artifact(s) for `umapka` go here.

- `model_core.pkl` — the gradient-boosted (LightGBM) regressor trained
  on 768-dim UMA difference features, as loaded by
  `PkaPredictor("models/model_core.pkl")` in the README and
  `examples/quickstart.py`. **Not included in this repository** — train
  it yourself (see below) or obtain it from the project maintainer, and
  place it here before running the examples.
- Do not commit large binaries to git without checking `.gitignore` /
  Git LFS — a `.pkl` regressor is typically small (KB–low MB) since UMA
  itself stays frozen and external, but confirm before adding it.

## Training

This directory currently has no training script. To reproduce
`model_core.pkl` you'll need to:

1. Build the 768-dim `[h_prot ; h_deprot ; h_prot − h_deprot]` feature
   for every molecule in the training set, via
   `PkaPredictor.features()` (this requires UMA / a GPU).
2. Fit a LightGBM (or other gradient-boosted) regressor against the
   experimental pKa labels from the
   [Machine-learning-meets-pKa](https://github.com/czodrowskilab/Machine-learning-meets-pKa)
   dataset (see top-level README's *Data* section).
3. `joblib.dump()` the fitted regressor to `model_core.pkl`.

If you want, a `train.py` script implementing this can be added under
`examples/` or a new `scripts/` directory.

---

Full project documentation lives in the top-level [README](../README.md).

Instead of computing deprotonation free energies (which we show does not
work — see below), this uses Meta's [UMA](https://huggingface.co/facebook/UMA)
universal atomistic model as a **frozen feature extractor**. Per-atom
embeddings are pulled from the input to UMA's energy head, pooled, and
combined as paired protonated/deprotonated difference features. A
gradient-boosted regressor maps those to pKa.

---

## Scope — please read before using

**Validated for:** monoprotic acids and bases, **pKa 2–12**. Also the
first ionization of simple polyprotic acids and bases in that range.

**Not reliable for:**

| Case | Behaviour |
|---|---|
| pKa₂ and beyond | MAE 1–3 units, frequently non-monotonic |
| Zwitterionic carboxyls (amino acids) | predicts ~5–8 where truth is ~2.2 |
| pKa below 2 | systematically over-predicted (no training data there) |
| pKa above 12 | unreliable (sparse training data) |

These are honest limitations, not bugs. See [Limitations](#limitations).

---

## Performance

| Evaluation | UMA embeddings | ECFP4 fingerprints |
|---|---|---|
| 5-fold CV (n = 5360) | **0.673** | 0.819 |
| **Scaffold split** (n = 1072, zero core overlap) | **0.994** | 1.090 |
| Novartis external (n = 263) | **1.170** | 1.445 |
| AvLiLuMoVe external (n = 122) | **0.696** | 0.721 |

All values are MAE in pKa units. Published random-forest benchmark on the
same dataset: 0.682 ([Baltruschat & Czodrowski 2020](https://f1000research.com/articles/9-113)).

**Scaffold split (0.994) is the honest headline number** — test molecules
share no Bemis–Murcko core with anything in training. Random-split
numbers are optimistic because analogues leak across the boundary.

### Validation against memorization

- **Label scrambling** collapses performance to the mean-prediction
  baseline (2.193 vs 2.101), confirming no information leakage.
- **Error depends only weakly on training similarity** (r = −0.155). For
  molecules with no close analogue (Tanimoto < 0.3), MAE is 1.162 against
  a 2.101 baseline.
- On molecules **verified absent from training**, designed chemical
  series are reproduced: inductive decay across five haloalkanoic acids
  (Spearman ρ = 1.000), Hammett ordering on substituted benzoic acids,
  and phenol substituent effects (ρ = 1.000).

---

## Install

```bash
git clone https://github.com/YOURUSERNAME/umapka.git
cd umapka
pip install -e .
```

UMA weights require a HuggingFace account with access to
[`facebook/UMA`](https://huggingface.co/facebook/UMA):

```bash
huggingface-cli login
```

A GPU is strongly recommended (~0.35 s per molecule; far slower on CPU).

---

## Usage

```python
from umapka import PkaPredictor

p = PkaPredictor("models/model_core.pkl")

p.predict("CC(=O)O")                    # acetic acid   -> ~4.2
p.predict("Oc1ccccc1")                  # phenol        -> ~10.0
p.predict("CCN")                        # ethylamine    -> ~10.7

# choose a specific site on a multi-site molecule
for s in p.sites("CC(=O)Nc1ccc(O)cc1"):
    print(s["index"], s["group"], p.predict_site("CC(=O)Nc1ccc(O)cc1", s["index"]))
```

---

## How it works

1. **Enumerate the titratable site** via SMARTS (after neutralizing —
   public datasets often store molecules already ionized).
2. **Build both charge states**, differing by exactly one proton.
3. **Extract UMA embeddings** for each: a forward pre-hook on the energy
   head captures the 128-dimensional per-atom representation before it is
   collapsed to a scalar energy.
4. **Pool** — L2-normalize per atom, then concatenate mean and max.
   Normalization matters; raw means are dominated by a few
   high-magnitude atoms.
5. **Concatenate** `[h_prot ; h_deprot ; h_prot − h_deprot]` (768-dim).
   pKa describes a *transition*, so both states and their difference are
   encoded. The difference term also cancels contributions from atoms far
   from the titrating site, which is why accuracy does not degrade with
   molecular size (tested to 41 heavy atoms).
6. **Predict** with a gradient-boosted regressor.

UMA itself is never fine-tuned.

---

## Why not compute the energies directly?

We tried. A full thermodynamic pipeline — Boltzmann-averaged over all
protonation microstates and conformers, with implicit solvation — fails
completely: no variant achieved positive R².

The reason is quantitative. One pKa unit corresponds to
**2.303·kT = 59.2 meV** at 298 K, so the entire 6.96-unit experimental
range spans only **412 meV**. Computed ΔG scatter spans **1046–1437 meV**,
2.5–3.5× the whole signal. UMA's own reported error on charged species
(200–500 meV) equals 3.4–8.5 pKa units.

This is not specific to UMA: it reproduces GFN2-xTB energetics closely
(r = 0.92 gas, 0.92 solvated), and both fail identically. It is a
precision wall shared across methods — which is why leading pKa
predictors train on experimental labels rather than computing from
energies.

---

## Limitations

- **Aqueous only.** No solvent is modelled; aqueous behaviour is learned
  entirely from the training labels. Will not transfer to other solvents.
- **Site selection is heuristic.** SMARTS matching covers ~89% of
  drug-like molecules; primary sulfonamides, N-oxides and weakly basic
  aryl amines are missed. For multi-site molecules the first match by
  priority order is used unless you call `predict_site` explicitly.
- **Single conformer** per structure, no ensemble averaging.
- **Polyprotic support has a separate, experimental model — read the
  numbers before trusting it.** `predict_chain()` (requires a
  `free_energy_model_path` at construction, trained via
  `train_free_energy_model.py`) predicts sequential multi-site
  deprotonation by scoring single-state UMA embeddings and taking
  differences, rather than predicting each transition's pKa directly -
  this avoids the earlier approach's failure mode (see below), but it
  is meaningfully less accurate than the main single-site model:
  scaffold-split MAE ≈1.47 (vs 0.994 for `model_core.pkl`), and on
  held-out multi-step chains the predicted pKa sequence comes out
  correctly ordered 88.9% of the time for 2-step chains, 61.9% for
  3-step (small sample, n=21). It does **not** sort results to force
  monotonicity - check the `"monotonic"` key in its output rather than
  assuming the order is right. Known hard cases from direct inspection
  of failures: near-degenerate true pKa gaps (<0.5 apart - often not a
  meaningful ordering to begin with), triprotic amino-diacids, and
  fused heteroaromatic systems with coupled tautomers.
  **This method's underlying code has been syntax-checked but not
  run-tested end-to-end** (written in an environment without UMA/rdkit
  access) - verify it works before relying on it.
- **`model_sequential_fixed.pkl` / `model_sequential_corrected.pkl` are
  abandoned dead ends, not usable artifacts, if you still have them
  from an earlier notebook export.** Investigation found the "corrected"
  version's feature-construction code silently falls back to unrelated
  data (it looks for `A_mol`/`B_mol` molecule objects that don't exist
  in the saved `feat_seq.pkl`), and its prediction wrapper forced
  ordering via `sorted(raw_predictions)` rather than fixing the
  underlying model - exactly the failure mode `predict_chain()` above
  was built to avoid. The notebook's own comments describe the
  immediately preceding version as producing "flat predictions around
  ~6-8, MAE ~2.94" - i.e. broken. Don't resurrect these.
- **Through-resonance substituents.** Nitro groups on phenols and
  anilines are mis-handled (up to ~2 pKa units), while nitro on benzoic
  acids is fine — the model captures induction better than resonance
  with the ionizable centre.

---

## Data

Training and evaluation use the MIT-licensed experimental pKa
compilation from the Czodrowski group:
[Machine-learning-meets-pKa](https://github.com/czodrowskilab/Machine-learning-meets-pKa)
(ChEMBL25 and DataWarrior sources; Novartis and AvLiLuMoVe held out as
external test sets).

---

## Citation

```bibtex
@misc{umapka2025,
  title  = {umapka: pKa prediction from UMA foundation-model embeddings},
  author = {[YOUR NAME]},
  year   = {2025},
  url    = {https://github.com/YOURUSERNAME/umapka}
}
```

Please also cite [UMA](https://arxiv.org/abs/2506.23971) and the
[dataset source](https://f1000research.com/articles/9-113).

## License

MIT
