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
