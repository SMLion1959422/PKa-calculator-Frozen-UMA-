# umapka

pKa prediction from **UMA foundation-model embeddings**.

Instead of computing deprotonation free energies (which we show does not
work — see below), this uses Meta's [UMA](https://huggingface.co/facebook/UMA)
universal atomistic model as a **frozen feature extractor**. Per-atom
embeddings are pulled from the input to UMA's energy head, pooled, and
combined as paired protonated/deprotonated difference features. A
gradient-boosted regressor maps those to pKa.

> **v3 methodology (this branch).** Three changes, documented in
> [`METHODOLOGY.md`](METHODOLOGY.md) and motivated in
> [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md):
>
> 1. **Thermodynamic microstate layer** (`predict_macro`, `--macro`).
>    Per-site predictions are coupled electrostatically and summed over
>    every protonation microstate, giving macro pKas, pI, titration
>    curves and species at any pH, temperature and ionic strength. The
>    coupling is validated independently of the ML model (LOO MAE 0.32
>    vs 1.01 for independent sites) and fixes zwitterions: for glycine,
>    pI 6.06 vs 6.06 experimental.
> 2. **Site assignment.** The SMARTS priority rule featurized the wrong
>    site, or no site, for 23% of the Novartis set; tetrazoles were never
>    deprotonated. Training now uses the annotated site.
> 3. **Site-local representation.** Embeddings are pooled around the
>    ionizable group, with UMA's ΔE as an extra feature. In an RDKit-only
>    ablation this takes Novartis MAE from 1.48 to 1.08 (fully automatic).
>    The UMA retrain (`dev/train_site_model.py`) still has to be run on a
>    GPU.
>
> **Benchmark** (`dev/benchmark_multiprotic.py`): on 26 amino acids and
> polyprotic molecules the microstate layer cuts pKa MAE from 1.40 to
> 0.84 (p = 2e-6) with the same per-site inputs. On the SAMPL6/SAMPL7
> blind sets it is neutral, and per-site accuracy (MAE 1.5–2.0) is the
> bottleneck. See `METHODOLOGY.md` §4.

---

## Scope — please read before using

**Validated for:** monoprotic acids and bases, **pKa 2–12**. Also the
first ionization of simple polyprotic acids and bases in that range.

**Not reliable for:**

| Case | Behaviour |
|---|---|
| pKa₂ and beyond | MAE 1–3 units, frequently non-monotonic — use `predict_macro` |
| Zwitterionic carboxyls (amino acids) | `predict()` gives ~5–8 where truth is ~2.2; `predict_macro` couples the sites and recovers the zwitterion |
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

### Multi-site, pH, temperature and ionic strength

```python
out = p.predict_macro("NCC(=O)O", pH=7.4, T_K=310.15,
                      salt="NaCl", salt_concentration=0.15,
                      return_model=True)
out["macro_pKas"], out["isoelectric_point"], out["dominant_species"]
out["model"].titration_curve()      # net charge / uncharged fraction vs pH
```

```bash
python predict_pka.py "NCC(=O)O" --macro --pH 7.4 --temperature 37 \
    --salt NaCl --molarity 0.15 --titration glycine.csv
```

Caveat: the ionic-strength term assumes the per-site pKas refer to
I ≈ 0. The trained models learn from literature values measured at
mixed ionic strengths, so `salt=` can double-count about 0.1 unit
(METHODOLOGY.md §4).

---

## HPLC / LC-MS buffer selection

A pKa is most useful when it answers *what pH should I run at, and what
buffer holds it there*. `umapka.buffers` derives that from the predicted
macro pKas:

```bash
python predict_pka.py "OC(=O)c1ccccc1" --buffer                  # LC-MS (volatile only)
python predict_pka.py "CCN" --buffer --uv --column hybrid-BEH    # UV; wider pH column
```

```python
from umapka.buffers import advise_smiles, recommend, format_report
advise_smiles(p, "OC(=O)c1ccccc1")["report"]     # predict pKa, then advise
print(format_report([4.20], 0, ms=True))          # or start from known pKas
```

Three rules, all standard practice, are applied together:

1. **Reproducibility** — stay ~2 pH units from every pKa, where the analyte
   is >=99% one species; near a pKa the ionized fraction swings steeply and
   retention moves with small pH errors.
2. **Buffer capacity** — a buffer only buffers within ~±1 unit of its *own*
   pKa. Outside that it is salt.
3. **Column and detector limits** — silica C18 is pH 2–8, hybrid particles
   wider; LC-MS needs volatile buffers, so phosphate and citrate are excluded
   unless you pass `--uv`.

Speciation uses the exact binding polynomial, so polyprotic analytes are
handled properly rather than one pKa at a time. The output reports the
tension the rules create rather than hiding it — a basic analyte that needs
pH 11.5 to run uncharged will be told so, along with which column reaches
it — and **zwitterions are distinguished from genuinely uncharged
molecules**, since net charge zero does not imply good reversed-phase
retention for an amino acid.

Not a replacement for method development: organic modifier, temperature and
stationary phase are not modelled.

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
(r = 0.92 gas, 0.92 solvated), and both fail identically.

**Caveat (v3):** this is true of *absolute* ΔG. Successful QM pKa
schemes use ΔG *relative to a reference in the same functional group*
or with a per-group linear fit (Klamt 2003; Jensen 2017; Fujiki 2018 —
see `LITERATURE_REVIEW.md` §2.1), where the systematic error cancels.
The v3 site features therefore include UMA's ΔE as one feature for the
regressor to calibrate per chemotype, not as the prediction itself.

---

## Limitations

- **Solvents.** The aqueous model learns water entirely from labels.
  Other solvents go through `multisolvent_tuned.pkl` (8 solvents in
  training), which interpolates but does not extrapolate to unseen
  solvent classes (RESULTS.md, leave-one-solvent-out).
- **Site selection is heuristic.** At inference the first SMARTS match
  by priority is used unless you call `predict_site` explicitly. It
  disagrees with the dataset's annotated site for 23% of Novartis
  molecules (`results/site_ablation.txt`); site-annotated training
  (`dev/train_site_model.py`) reduces but does not remove the cost.
- **Single conformer** per structure, no ensemble averaging.
- **Polyprotic support:** `predict_macro` is the supported route (coupled
  microstates; coupling validated on diacids/diamines, see
  `METHODOLOGY.md`). The older `predict_chain` notes follow.
  Second ionizations require
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
