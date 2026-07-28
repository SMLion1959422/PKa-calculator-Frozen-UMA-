# umapka

pKa prediction from **UMA foundation-model embeddings**.

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
git clone https://github.com/SMLion1959422/umapka.git
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

## Solvents, molarity, and solvent mixtures

See [`MERGE_NOTES.md`](MERGE_NOTES.md) for what changed and why. Short
version:

```bash
# pure non-aqueous solvent
python predict_pka.py "CC(=O)O" --solvent dmso

# add a salt at a given molarity (ionic-strength correction)
python predict_pka.py "CC(=O)O" --salt NaCl --molarity 0.15

# binary solvent mixture
python predict_pka.py "CC(=O)O" --mix water:acetonitrile --fraction 0.3

# reference lists
python predict_pka.py --list-solvents
python predict_pka.py --list-salts
```

Or from Python:

```python
from umapka import PkaPredictor
from umapka.mixtures import predict_mixed_solvent_pka

p = PkaPredictor("models/model_core_v2.pkl")

p.predict("CC(=O)O", solvent="dmso")
p.predict("CC(=O)O", salt="NaCl", salt_concentration=0.15)
predict_mixed_solvent_pka(p, "CC(=O)O", "water", "acetonitrile", fraction_b=0.3)
```

**Why molarity is a calculation, not a trained model.** `salt=` /
`salt_concentration=` apply a physics-based ionic-strength correction
(Debye-Hückel/Davies, extended with a Bjerrum ion-pairing term for
non-aqueous solvents — see `umapka/solvation.py`) on top of the
*trained* base pKa prediction. This is intentional, not a shortcut:
concentration-dependent pKa *shift* data (same molecule, same solvent,
several ionic strengths) is much rarer than plain pKa data, and this
classical theory is genuinely well-validated for dilute aqueous
solutions — there's no reason to prefer an undertrained ML correction
over settled 20th-century electrochemistry here. If you have real
concentration-dependent shift data, training a *residual* correction
on top of this physics-based estimate (rather than replacing it) would
be the way to improve it further — see `MERGE_NOTES.md`.

**Why mixtures are endpoint-anchored, not fed to the ML model
directly.** No solvent *mixture* is in either branch's training data.
Feeding the regressor a made-up "interpolated dielectric constant"
would look reasonable (the model does take a continuous epsilon
feature) but is silent extrapolation — a tree-based regressor doesn't
degrade gracefully outside its training range. Instead,
`predict_mixed_solvent_pka` predicts the two *pure*-solvent endpoints
with the trained model (the part training is actually validated for),
then interpolates between them using the Yasuda–Shedlovsky relation
(pKa approximately linear in 1/epsilon) — the same technique used in
real pharmaceutical pKa determination by cosolvent extrapolation. It
reports a `confidence` field and a `warning` when you're outside the
water-rich composition range that relation is best-established for.
See `umapka/mixtures.py` for the full derivation and caveats.

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
- **Polyprotic support is not validated.** Second ionizations require
  −1 → −2 transitions, which are absent from the training distribution.
  Public data is thin here (~1,348 polyprotic molecules across both
  source datasets, of which only ~67 yield usable charge−1 transitions),
  and attempts to correct for it did not reproduce reliably. This is the
  main open problem.
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
  author = {Srikanth Mohan},
  year   = {2025},
  url    = {https://github.com/SMLion1959422/umapka}
}
```

Please also cite [UMA](https://arxiv.org/abs/2506.23971) and the
[dataset source](https://f1000research.com/articles/9-113).

## License

MIT
