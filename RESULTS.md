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

## 6. v3 methodology results (see METHODOLOGY.md)

**Microstate coupling** (no ML involved; `results/coupling_validation.txt`):
pKa2 - pKa1 of 17 symmetric diacids/diamines, LOO MAE **0.32**
(independent sites 1.01, constant gap 0.80). Glycine, held out:
macro 1.81 / 10.32 (exp 2.35 / 9.78), pI 6.06 (exp 6.06).

**Site assignment** (`results/site_ablation.txt`): SMARTS priority
disagrees with the annotated site for 22.9% of Novartis, 14.9% of
training, 0% of AvLiLuMoVe.

**RDKit-only ablation**, same LightGBM:

| arm | Novartis | AvLiLuMoVe |
|---|---|---|
| global Morgan | 1.48 | 0.77 |
| + site-centred @ SMARTS site | 1.15 | 0.59 |
| trained @ annotated sites, SMARTS site at test | **1.08** | **0.53** |
| + site-centred @ annotated site | 0.99 | 0.53 |

**UMA site_v3 retrain** (`dev/train_site_model.py`, Colab T4, fairchem-core 2.23,
5980/5994 training molecules featurized; single run):

| | old UMA model_core_v2 | RDKit site baseline | **UMA site_v3** |
|---|---|---|---|
| scaffold 5-fold CV | - | - | 0.72 |
| Novartis, SMARTS site (automatic) | 1.17 | 1.08 | **1.03** |
| Novartis, annotated site | - | 0.99 | **0.86** |
| AvLiLuMoVe | 0.70 | 0.53 | **0.43** |

UMA features beat the RDKit baseline on every split (-13% Novartis annotated,
-18% AvLiLuMoVe), so the foundation-model representation adds information once it
is pooled around the site. Remaining Novartis gap is site choice (0.86 vs 1.03).

**Multi-pKa benchmark** (`results/benchmark_multiprotic.txt`; same intrinsic
pKas for both arms, benchmark molecules removed from training):

| set | independent sites | microstate layer |
|---|---|---|
| amino acids + polyprotic, 60 pKas | 1.40 | **0.84** (p = 2e-6; pI 0.57 -> 0.38) |
| SAMPL6, 31 pKas | 1.52 | 1.51 (spurious pKas 26 -> 12) |
| SAMPL7, 20 pKas | 1.99 | 2.05 (ionic-strength term; 2.00 without) |

**Multi-pKa benchmark with UMA site_v3 intrinsic pKas**
(`results/benchmark_multiprotic_uma.txt`; microstate layer, MAE):

| set | rdkit_site | **UMA site_v3** | independent sites (UMA) |
|---|---|---|---|
| amino acids + polyprotic | 0.84 | **0.67** (pI 0.29, charge@7.4 25/26) | 1.59 |
| SAMPL6 | 1.51 | **0.94** | 1.06 |
| SAMPL7 | 2.05 | **1.26** | 1.22 |

UMA cuts SAMPL error by ~38% vs the RDKit site model. With UMA inputs the
ionic-strength term helps on both SAMPL sets (0.94 vs 1.10 at I=0; 1.26 vs 1.37).
Still ~2x behind Uni-pKa (0.49 / 0.55). Caveat: the UMA model was not retrained
with benchmark molecules excluded (3 overlaps, amino-acid set only).

**Automatic site choice** (`dev/site_choice.py`, `results/site_choice_rdkit.txt`):
the annotated site is the single site ChemAxon Marvin places in pKa 2-12
(dataset pipeline), so "window" rules were tested. With rdkit_site per-site
pKas on the 275 Novartis molecules that have a SMARTS site: priority 1.077,
window 1.077 (identical picks), window on coupled apparent pKas 1.217,
oracle 1.049. Choosing among detected sites is worth <= 0.03; most of the
UMA annotated-vs-automatic gap (0.86 vs 1.03) comes from sites the SMARTS
table does not cover at all. Next lever: SMARTS coverage (aromatic N-H acids,
amidines/heterocyclic bases), not site ranking.

