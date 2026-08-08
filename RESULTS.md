# UMA-pKa: Results Summary

## 1. Core idea

UMA's raw energetics fail on pKa (SAMPL6: R^2 = -0.44, inverse correlation).
But UMA's **learned embeddings**, fed to a trained LightGBM head, predict pKa
well in water and across solvents.

> UMA's energies are the wrong tool; its internal representations are the right one.

## 2. Aqueous model (model_core_v2)

**Leakage fix:** the shipped model_core.pkl scored 0.561 train / 0.606 test
(nearly identical) -- it saw its "held-out" data. All results below use
freshly trained, leakage-checked models.

| model | Novartis MAE | vs shipped |
|---|---|---|
| shipped model_core.pkl (leaky) | 1.41 | baseline |
| **v2 (clean retrain + calibration)** | **1.16** | -18% |

### Size dependence (key finding)

Error scales monotonically with molecule size on external data:

| size (heavy atoms) | Novartis MAE | AvLiLuMoVe MAE |
|---|---|---|
| < 15 | 0.68 | 0.46 |
| 15-22 | 1.01 | 0.60 |
| 22-30 | 1.08 | 0.64 |
| > 30 | 1.44 | 1.05 |

**Global mean-pooling dilutes local pKa signal on large molecules.**
A size-aware correction does NOT transfer across datasets -- the problem
is representational, not a correctable bias.

### Best/worst performance profile

Best: small, 1-ring, pKa 7-10 (MAE ~0.5-0.9)
Worst: large, 3+ rings, pKa <4 or >10 (MAE ~1.1-1.5)

## 3. Multi-solvent model (multisolvent_tuned)

Extended to 8 solvents using the Nevolianis et al. Anion Solvation dataset
(Zenodo 15604045, CC BY 4.0): 8,241 experimental pKa values.

### The aprotic fix

| solvent | before (no data) | after (8k dataset) |
|---|---|---|
| DMSO | 8.85 | **1.39** |
| Acetonitrile | 8.51 | **1.33** |
| DMF | 8.84 | **0.95** |

### Tuned held-out test (their published split)

**Overall MAE: 0.822 (untuned) -> 0.711 (tuned), -13.5%**

| solvent | test MAE | n |
|---|---|---|
| Ethanol | 0.18 | 25 |
| DMF | 0.40 | 30 |
| Methanol | 0.62 | 45 |
| Water | 0.64 | 364 |
| Acetonitrile | 0.71 | 33 |
| DMSO | 1.15 | 97 |

Published GNN multi-solvent models on same data: ~0.58 MAE.
UMA embeddings + LightGBM: 0.711 -- competitive with simpler machinery.

### !! The 0.711 above is NOT what a user gets

Every number in this section was produced by feeding the dataset's
**exact reaction pairs** (AH>>A-) straight in as features. That is an
oracle: it hands the model the correct titratable site. Real usage -
`predict(smiles, solvent=...)` - has to FIND the site itself.

`validate_multisolvent.py` measures the production path on the SAME
published held-out split:

| solvent | oracle-pair (above) | **production** | site agreement | n |
|---|---|---|---|---|
| Water | 0.64 | **1.50** | 78.5% | 364 |
| DMSO | 1.15 | **4.55** | 69.2% | 97 |
| Methanol | 0.62 | **0.84** | 90.7% | 45 |
| Acetonitrile | 0.71 | **2.12** | 78.1% | 33 |
| DMF | 0.40 | **1.43** | 83.3% | 30 |
| Ethanol | 0.18 | **0.25** | 92.0% | 25 |
| **overall** | **0.711** | **1.795** | 79.4% | 603 |

The gap tracks site agreement almost perfectly: solvents where the site
finder agrees with the dataset's own reaction (Ethanol 92%, Methanol
91%) are close to the oracle number; where it does not (DMSO 69%) the
error is 4x worse. Non-aqueous pKa spans a much wider range than
aqueous, so a wrong site costs far more there.

**Quote 1.795, not 0.711, as the multisolvent capability.** The 0.711 is
a legitimate measure of the regressor, but only under an assumption the
shipped tool cannot satisfy. NMP (n=4) and ethylene glycol (n=5) have
too few held-out points to claim anything at all.

Known bug: validate_multisolvent.py's exact-pair column returns all-NaN,
because it calls features() without site indices, which the 1536-dim
aqueous model requires. The production column is correct.

### Generalization boundary

- Random-split (solvent in training): ~1.0 MAE everywhere. Features carry signal.
- Leave-one-solvent-out: protic transfers (MeOH 1.10, EtOH 0.86); aprotic doesn't
  (DMSO 6.14, MeCN 9.40). Model interpolates, doesn't extrapolate.

## 4. Limitations

- Aqueous accuracy degrades on large/polycyclic/extreme-pKa molecules (global pooling).
- Multi-solvent can't extrapolate to unseen solvent classes.
- Site selection uses fixed-priority SMARTS; polyfunctional molecules may pick wrong site.
- All data are single pure solvents; mixtures out of scope.
- IUPAC aqueous data is CC BY-NC (non-commercial only).

## 5. Reproducibility

- Python 3.11 + fairchem-core (3.14 NOT supported)
- UMA embeddings cached; ~30 min to recompute 8k molecules on CPU
- Data: aqueous (ChEMBL/DataWarrior), external (Novartis, AvLiLuMoVe),
  multi-solvent (Zenodo 15604045, CC BY 4.0)

---

## 6. Self-contained pipeline: what moved Novartis and what did not

All numbers below are the REAL-WORLD configuration: site found by the
model itself, no ChemAxon annotations at inference.

### Read this before comparing datasets

`check_test_novelty.py`, median nearest-neighbour Tanimoto to training:

| set | median NN | >=0.9 (near-duplicate) |
|---|---|---|
| Novartis | 0.365 | 0.7% |
| AvLiLuMoVe | 0.708 | 27.6% |

**Novartis is genuine extrapolation; AvLiLuMoVe is largely
interpolation.** A change can improve one while making the other worse -
this happened repeatedly. Do not optimise the combined average.

### What worked - ALL of it was site assignment, none of it the regressor

| change | Novartis |
|---|---|
| baseline (v16 + learned site finder) | 1.001 |
| + learned acid/base kind classifier | 0.965 |
| + stochastic-bagged ensemble (v20) | 0.949 |
| + hard-negative ballot expansion | 0.944 |
| + kind classifier retrained on the expanded ballot | **0.918** |

**Ballot expansion** (`models/site_detector.pkl`, train_hard_negatives.py).
SMARTS only *enumerates* candidates, so a titratable atom no pattern
matches is unreachable however good the ranker is - 3.9% of Novartis,
6.0% of training. A classifier over every N/O/S/P atom, trained with
non-titratable heteroatoms (amides, esters, backbone links) as explicit
hard negatives and the pKa regression masked to positives, adds the
confident ones to the ballot; the ranker still picks. Measured with
rank/n_cands recomputed per arm (eval_union_hybrid.py): 94.2% -> 96.8%
site accuracy, **7 molecules recovered, 0 broken**. Primary sulfonamides,
thiols and phosphonic acids are now reachable.

A 35-dim variant with all 60 SMARTS one-hots REMOVED scored the same as
the 95-dim version (83.1% vs 83.4% recall on SMARTS-invisible sites),
confirming it learned chemistry rather than "a pattern matched me".

**Kind classifier retrain.** Widening the ballot initially made kind
agreement WORSE (96.7% -> 95.7%), because train_kind_classifier.py had
`if idx not in hits: return None` - it had silently dropped the 302
training molecules whose true site is not SMARTS-matched, i.e. exactly
the atom types the new ballot surfaces. Featurising them the way
predict_kind does at inference (empty pattern info, n_cands/prio_rank
over hits UNION {idx}) fixed it:

| | before | after |
|---|---|---|
| training molecules | 5636 | 5938 |
| Novartis kind, on the correct atom | 99.3% (2 wrong) | **100.0% (0 wrong)** |
| production kind agreement | 95.7% (12 wrong) | **98.6% (4 wrong)** |

Every Novartis size bucket improved and AvLiLuMoVe held exactly flat
(0.411), so this is a real gain on novel chemistry, not a distribution
shift like v21.

**Kind classifier** (`models/kind_classifier.pkl`) was the single biggest
win. Error decomposition of v16 on Novartis:

| | MAE | n |
|---|---|---|
| acid/base kind wrong | **3.901** | 14 |
| kind correct | 0.846 | 261 |

A wrong kind computes the OPPOSITE proton transfer, so the answer is
meaningless rather than imprecise. The classifier is 99.3% accurate held
out vs 94.9% for the H-count rule it replaced.

**Stochastic bagging**: LightGBM is deterministic at `subsample=1.0`, so
an earlier "3-seed bag" silently trained three byte-identical models
(all scoring OOF 0.4961). `subsample=0.8, subsample_freq=1,
colsample_bytree=0.8` makes seeding do anything at all.

### Negative results - do not repeat without new data

| hypothesis | outcome |
|---|---|
| xTB quantum descriptors | 0.541 vs 0.521 for free Gasteiger/EState; stacking both adds +0.001 |
| attention pooling over per-atom embeddings | 0.627 OOF vs 0.543 for a 3-model average |
| more training data (+6216 clean molecules) | Novartis 0.949 -> **1.075** |
| inverse-propensity size reweighting | Novartis 0.949 -> **0.971** |
| end-to-end UMA fine-tuning (GPU) | **no gain over frozen UMA** |
| higher-L (L=1/L=2) spherical-harmonic channels | +0.0080 OOF, 95% CI [-0.019, +0.036], p=0.574 |
| SASA solvent-exposure descriptors | -0.0020 OOF, 95% CI [-0.006, +0.002] |
| LoRA / PEFT adapters | **not applicable** - UMA has 0 attention modules |

**Higher-L channels.** UMA's backbone emits `(n_atoms, 9, 128)`; every
feature here uses the L=0 scalar block, discarding the L=1 dipole and
L=2 quadrupole channels. Their per-channel norms are genuine rotation
invariants - verified by rotating a molecule: raw L1/L2 components moved
1.19 and 4.33, their norms moved 2.0e-4 and 2.4e-5. Adding them changed
nothing (497 molecules improved, 471 worsened - a coin flip). The energy
head consumes L=0, so training has already concentrated
energetically-relevant information there; the anisotropy is real but
redundant for an energy-like target. It would likely help for a
genuinely anisotropic target (dipoles, NMR shifts).

**SASA.** Explicit solvent-exposure descriptors of the titratable atom
and its 1/2/3-bond shells. Null, and a CONFIDENT null - run on all 5184
molecules, the CI excludes any effect larger than +/-0.006. UMA consumes
3D coordinates directly, so steric shielding is already encoded in its
input; SASA is a re-encoding, not new information. (SASA as *pooling
weights* was deliberately not tested - that re-weights information
already present, and pooling variants have failed here three times.)

**LoRA.** Proposed as "inject LoRA into the attention blocks". Inspecting
the loaded backbone: `eSCNMDMoeBackbone`, module census
`{SiLU: 28, Linear: 26, MOLE: 24, SO2_m_Conv: 16, ...}`, **attention
modules: 0**. It is an equivariant spherical-channel network, not a
transformer. The 26 Linear layers are radial-basis MLPs and a mixing
layer; the geometric reasoning lives in SO2_m_Conv/MOLE, which are not
what LoRA wraps.

**Size reweighting (v22).** Training is 7.0% large molecules (>30 heavy
atoms) against 22.5% for Novartis, so examples were reweighted by
inverse propensity to match the target's size profile (weights: 0.17 for
<15 atoms, 3.13 for >30). Every bucket got worse, including the one it
was built to fix:

| size | v20 | v22 |
|---|---|---|
| <15 | 0.854 | 0.910 |
| 15-22 | 0.820 | 0.835 |
| 22-30 | 0.999 | 1.010 |
| >30 | **1.065** | **1.100** |

Together with v21 this is **two independent tests** of "the large-molecule
bucket needs more large-molecule emphasis" - adding them (v21) and
reweighting toward them (v22) - and both made that bucket WORSE. The
>30 error is therefore not a data-representation problem. It is
intrinsic: either global-pooling dilution on large structures (the
effect RESULTS.md section 2 documents) or genuinely harder chemistry.
Downweighting the 50.6% of training molecules under 15 atoms also
destroyed signal that was helping.

**More data (v21).** The added molecules are contamination-free (0 exact
overlap with either test set) and 3x richer in large molecules
(21.6% vs 7.0% >30 atoms), yet EVERY Novartis size bucket got worse:

| size | v20 | v21 |
|---|---|---|
| <15 | 0.854 | 1.049 |
| 15-22 | 0.820 | 0.870 |
| 22-30 | 0.999 | 1.061 |
| >30 | **1.065** | **1.371** |

They are size-matched to Novartis but scaffold-matched to AvLiLuMoVe
(median NN Tanimoto 0.643 vs 0.330), so AvLiLuMoVe improved 31%
(0.411 -> 0.284) while Novartis degraded. This reproduces and explains
v18 "maxdata". `models/model_core_v21_maxdata.pkl` is kept but should
NOT be promoted for Novartis-like chemistry.

**Fine-tuning UMA** (Colab T4, 5184 molecules, blocks 3-4 unfrozen,
two-phase: head warmup then unfreeze):

```
PHASE A  head only, UMA frozen : 1.375 -> 1.224 -> 1.130   (-0.151, -0.094/ep)
PHASE B  blocks 3,4 unfrozen   : 1.103 -> 1.070 -> 1.050   (-0.027, -0.033, -0.020/ep)
```

Phase A improved 3x faster with UMA **completely frozen**. Phase B's
decline matches simple deceleration of head-only training, so unfreezing
bought approximately nothing. Train 0.658 vs val 1.050 = memorising.
Fine-tuned Novartis 1.622 vs frozen 0.949.

Two fairchem gotchas found doing this, both silent:
- `get_potential_energy()` computes forces, whose internal backward
  FREES the graph - a pKa loss then dies with "backward through the graph
  a second time". Call `backbone(...)` directly instead.
- `AtomicData.from_ase(...)` **drops charge** unless given
  `r_data_keys=["charge","spin"]`. Same geometry at charge 0 vs -1
  differed by 3.6e-07 (noise) without it, 3.9e-01 with it. pKa is a
  charge-changing transition, so a model missing this is worthless while
  looking perfectly healthy.

### The wall

Final error decomposition on Novartis (v20 + expanded ballot + retrained
kind classifier, MAE 0.918):

| | MAE | n |
|---|---|---|
| site AND kind both correct | **0.841** | 274 |
| kind wrong | 6.156 | 4 |
| >30 heavy atoms | 1.039 | 63 |
| 15-22 heavy atoms | 0.784 | 83 |

Fixing the last 4 mismatches would yield ~0.077, landing at 0.841 - which
is also the oracle-site number, so site selection cannot pass it.
**Site assignment is now effectively solved** (98.6% kind agreement,
previously-unreachable chemistry now on the ballot); everything left is
the regressor.

### The pattern, across the whole session

**Every approach targeting SITE ASSIGNMENT worked. Every approach
targeting the REGRESSOR failed.**

    site assignment : 1.001 -> 0.918   (kind classifier, bagging,
                                        ballot expansion, kind retrain)
    regressor       : 8 approaches, all null or negative

Five separate OOF improvements produced zero or negative Novartis
movement. Training data and AvLiLuMoVe are near neighbours (median NN
Tanimoto 0.708) while Novartis is novel (0.365), so cross-validated fit
on training chemistry is the wrong objective for it. This is a
GENERALISATION gap, not a fitting gap, and the binding constraint is
5184 training molecules, not model capacity, feature engineering, or
architecture.

**Do not attack the regressor again without new data.** Specifically,
the two data strategies NOT yet tried, and the reason each differs from
what already failed:
  - targeted curation of complex/macrocyclic/polyheterocyclic molecules
    with EXPERIMENTAL pKa (v21 failed because its molecules were
    scaffold-matched to AvLiLuMoVe, not to Novartis - size matching is
    not enough)
  - high-level QM (DFT-class) reference data (cheap semi-empirical xTB
    already measured NEGATIVE, so a semi-empirical proxy is not a
    substitute)

For reference: ChemAxon Marvin scores 0.856 on Novartis **using its own
site annotations**; 0.949 here is fully self-contained.

### Environment pins that matter

Two version mismatches caused silent, expensive failures:

- **lightgbm**: a 3.3.5 pickle raises inside `.predict()` under 4.x
  (`_n_classes` is None). `site_finder.py` swallowed that into a
  SMARTS-priority fallback, so the learned site finder appeared to work
  while never running. It now warns loudly instead.
- **scikit-learn**: installed 1.3.2 vs the locked 1.9.0 made LightGBM
  `.fit()` die with a Windows access violation. Predictions were
  bit-identical after fixing, so the calibrators were fine - but the
  crash was real.
