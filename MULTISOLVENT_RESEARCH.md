# Predicting pKa in different solvents: what works, and a plan for umapka

This covers pure non-aqueous solvents and water–cosolvent mixtures. It
reviews the approaches in the literature, diagnoses why umapka's current
multisolvent model fails on unseen solvents, and ranks concrete changes.

---

## 1. Why this is a different problem from aqueous pKa

pKa in solvent S is the free energy of HA → A⁻ + H⁺ **in S**. Split it
with a thermodynamic cycle against water (W):

```
pKa(S) = pKa(W) + [ΔG_tr(A⁻) − ΔG_tr(HA) + ΔG_tr(H⁺)] / (RT ln10)
```

ΔG_tr(X) = G_solv(X, S) − G_solv(X, W) is the transfer free energy of X
from water to S. Three physical effects dominate:

1. **Anion solvation.** Water stabilizes anions strongly through H-bond
   donation; aprotic solvents such as DMSO and acetonitrile cannot. This
   is why acetic acid goes from pKa ≈ 4.8 in water to ≈ 12 in DMSO and
   ≈ 23 in acetonitrile.
2. **Proton solvation**, ΔG_tr(H⁺). It is one constant per solvent, but
   a large one. It is what makes DMSO and MeCN scales differ by roughly
   10 units, although neither solvent donates H-bonds.
3. **Charge type.** For bases (BH⁺ → B + H⁺) the charges on the two
   sides are balanced, so the shifts between solvents are much smaller
   than for neutral acids.

**Consequence:** the solvent shift is *large* (up to about 20 units),
*solute-dependent* (it depends on how well the anion's charge is
delocalized) and *solvent-specific*. Two numbers per solvent, ε and a
protic flag, cannot encode it. That is what the current model does.

---

## 2. Approaches in the literature

| # | Approach | Key reference | Accuracy reported | Unseen solvents? | Cost |
|---|---|---|---|---|---|
| A | **Empirical conversion (LFER)**: pKa(S) = pKa(W) + c(S, family), with one additive constant per solvent × functional-group family | Rossini, Bochevarov & Knapp, *ACS Omega* 2018 ([doi](https://doi.org/10.1021/acsomega.7b01895)) | often ≈ 0.5 units (water → MeCN, DMSO, MeOH) | no (needs c fitted per solvent) | trivial |
| B | **Reference-solvent + COSMO-RS transfer**: the cycle above, with ΔG_tr from COSMO-RS and ΔG_tr(H⁺) fitted as the single parameter per solvent | Zheng, Al Ibrahim, Kaljurand, Leito & Green, *J. Comput. Chem.* 2025, 46, e27517 ([doi](https://doi.org/10.1002/jcc.27517)) | < 1 unit in 6 of 10 solvents; worse for large molecules | **yes**, given one calibration point | a COSMO-RS calculation per species (commercial) |
| C | **Multisolvent GNN on reaction SMILES + solvent** | Nevolianis et al., *JACS* 2025 ([doi](https://doi.org/10.1021/jacs.5c02578)); the same 8,241-point, 8-solvent data umapka uses | **MAE 0.58** on unseen solutes; 0.59 on SAMPL7 | only the 8 training solvents | cheap |
| D | **"Holistic" ML with physical-organic descriptors** (SPOC) over 39 solvents from iBonD | Yang et al., *Angew. Chem. Int. Ed.* 2020, 59, 19282 ([doi](https://doi.org/10.1002/anie.202008528)) | MAE 0.87 | partly (solvent as a feature) | cheap |
| E | **ML solvation free energies with physically motivated descriptors**, then the cycle | 2025 study, PubMed [40627128](https://pubmed.ncbi.nlm.nih.gov/40627128/); Vermeire & Green 2021 transfer learning (SolProp) ([arXiv](https://arxiv.org/abs/2012.11730)) | ΔG_solv MAE 0.44 / 1.72 / 1.60 kcal/mol for neutrals / anions / cations; 1 pK ≈ 1.36 kcal/mol | **yes**, if solvent descriptors are continuous | cheap |
| F | **Mixtures**: family-specific linear pKa(mixture) vs pKa(W) relations with preferential solvation | Rosés, Bosch and co-workers: methanol–water (*Anal. Chim. Acta*) and acetonitrile–water (*J. Chromatogr. A*, "Retention of ionizable compounds in HPLC" series), about 2001–2002 | good in the water-rich region | not applicable | trivial |
| G | **Mixtures by ML**: pooled solvent embeddings (MolPool / SolProp-mix) | [arXiv 2412.01982](https://arxiv.org/abs/2412.01982) | ΔG_solv in binary and ternary mixtures | yes, for mixture compositions | cheap |
| H | **Explicit / implicit-solvent ML potentials** (e.g. ConSolv, [arXiv 2606.24983](https://arxiv.org/abs/2606.24983)) | – | research stage | in principle | expensive |

**Consensus in the field:**
* Do **not** learn pKa(S) from scratch per solvent. Anchor to a
  data-rich reference (water, or DMSO) and model the **shift**, which is
  smoother and physically structured (A, B, E).
* Represent solvents by **continuous physical descriptors**: ε,
  Kamlet–Taft α (H-bond donation), β (H-bond acceptance) and π*
  (polarizability), E_T(30), Abraham parameters, or COSMO σ-profiles.
  These, not a solvent identity or a protic flag, are what allow
  prediction in solvents absent from training. A 2025 benchmark
  ([arXiv 2512.19530](https://arxiv.org/abs/2512.19530)) makes the same
  point with leave-one-solvent-out tests. Descriptors transfer to new
  solvents, while learned solvent embeddings do not.
* Treat **acids and bases separately** (different charge types, §1).
* The **proton term** is one number per solvent. Fit it rather than
  predict it (B).

---

## 3. Diagnosis of umapka's multisolvent model

`multisolvent_tuned.pkl`: LightGBM on [UMA pair embedding (768) ; ε/78.4 ; protic ∈ {0, 0.5, 1}].

| Evaluation (RESULTS.md) | MAE |
|---|---|
| Nevolianis test split (solvents all seen) | 0.71 (Nevolianis GNN: 0.58) |
| Leave-one-solvent-out, MeOH / EtOH | 1.10 / 0.86 |
| Leave-one-solvent-out, **DMSO / MeCN** | **6.14 / 9.40** |

What is wrong, in order of importance:

1. **The target is the absolute pKa(S), not the shift from water.** The
   model must re-learn the whole solute chemistry in every solvent from
   about 100–400 points each, instead of re-using the aqueous model
   (trained on thousands of points; Novartis MAE 1.03 / 0.86 with
   site_v3).
2. **The solvent is encoded by two coarse numbers.** DMSO (ε 46.7,
   Kamlet–Taft β ≈ 0.76) and MeCN (ε 37.5, β ≈ 0.3–0.4) look almost identical to the model
   (0.60/0 vs 0.48/0), yet their pKa scales differ by about 10 units,
   mostly through the proton term. Unseen aprotic solvents therefore
   fail by 6–9 units. That is not noise; the information is simply
   missing.
3. **A tree model cannot extrapolate** in ε. Any solvent outside the
   training ε range falls back to the nearest leaf.
4. **It uses whole-molecule pooling**, which site_v3 has since shown is
   the weaker representation.
5. **Acids and bases share one model** although their solvent shifts
   behave differently.

---

## 4. Recommended plan for umapka (ranked by value per effort)

### Step 1: Δ-learning on top of the aqueous model (highest value)

```
pKa(S) = pKa_W(site_v3)  +  c(S, charge type)  +  f( h_site , d(S) )
          aqueous model     per-solvent offset     learned residual shift
```

* `pKa_W` comes from the best aqueous model. Where the Nevolianis data
  have a measured water value for the same solute, use it during
  training.
* `c(S, charge type)` is the Rossini-style offset: a per-solvent
  constant for acids and one for bases, fitted by least squares. It
  absorbs the proton term.
* `f` is a small regressor on site-local UMA features (`pool_site`),
  plus UMA's anion ΔE and a **continuous solvent descriptor vector
  d(S)**: ε, 1/ε, α, β, π*, E_T(30) and the proton-affinity-type offset.
  It learns the solute-dependent part: charge delocalization and H-bond
  capacity of the anion.
* **Evaluation protocol** (report all three):
  1. the Nevolianis split, to compare with their GNN's 0.58;
  2. **leave-one-solvent-out**, the metric that matters, currently
     6–9 units for aprotic solvents;
  3. SAMPL7 non-aqueous, if available.
* **Baselines**, each also a useful product on its own:
  (i) `c(S, family)` alone, i.e. Rossini ECM;
  (ii) (i) plus the aqueous model.

  If the learned residual does not beat (ii), ship (ii).

### Step 2: Continuous solvent descriptors, enabling new solvents

Add a solvent table covering the 8 training solvents and about 20
common others (THF, acetone, DCM, DMF, NMP, pyridine, alcohols, ...)
with ε, Kamlet–Taft α/β/π* and E_T(30). These are tabulated in Marcus's
and Reichardt's compilations. For a new solvent the only unknown is
c(S), which the Zheng/Green result shows can be fitted from **a single
measured pKa** in that solvent. Expose it as
`predict(smiles, solvent=..., calibrate_with=[(smiles, pKa)])`. That
turns "unsupported solvent" into "one measurement away", which is
genuinely useful to experimentalists.

### Step 3: Mixtures, validated

Keep the endpoint-anchored Yasuda–Shedlovsky scheme, but
**validate it** against published water–MeOH and water–MeCN pKa series
(Rosés & Bosch). Replace it with their family-specific linear
relations where those fit better. Report errors against composition.

### Step 4: Microstates in other solvents

`umapka.microstates` already takes ε. In low-ε solvents the site-site
coupling grows sharply, since W ∝ 1/ε. Diacids in DMSO show very large
ΔpKa, which is a direct test once solvent pKa₁/pKa₂ data are collected.

### What not to do

* Do not add more solvent flags or identity one-hot encodings. They
  cannot generalize by construction.
* Do not interpolate the tree model in ε for mixtures. It is already
  documented as unsafe in `umapka/mixtures.py`.

---

## 5. Data sources

| Data | Content | Licence |
|---|---|---|
| Nevolianis et al., Zenodo 15604045 | 8,241 pKa in 8 solvents + COSMO-RS ΔG_solv of anions/neutrals + DLPNO gas-phase acidities | CC BY 4.0 |
| Zheng/Green supporting data, Zenodo 11153563 | pKa in 10 solvents with COSMO-RS transfer energies | see record |
| iBonD (Cheng group) | about 39 solvents, the largest compilation | web access |
| Tartu (Leito) acidity/basicity scales | high-quality MeCN, DMSO, DCE, heptane scales | literature |
| Rosés & Bosch series | pKa in MeOH–water and MeCN–water mixtures | literature |

---

## 6. Step 1 experiment: does shift modelling generalize to unseen solvents?

`dev/solvent_shift_experiment.py` tests the hypothesis:

> Can a frozen UMA representation predict solvent-induced pKa **shifts**
> better than predicting absolute pKa independently in each solvent?

| Model | Formula | Role |
|---|---|---|
| M1 | pKa_S = f(solute, ε/78.4, protic) | current formulation, with the same site features as M4 |
| M1b | pKa_S = f(solute, descriptors) | control: descriptors without the shift formulation |
| M2 | pKa_S = pKa_W + Δ_S (per solvent and charge type) | simple offset baseline |
| M3 | pKa_S = pKa_W + Ridge(descriptors, descriptors × pKa_W) | physics-style shift |
| **M4** | M3 + LightGBM(site features, descriptors, pKa_W) | proposed |

**The key comparison is M2 vs M4.** Splits:
* random, grouped by solute;
* **leave-one-solvent-out (the centrepiece)**;
* leave-one-chemical-family-out;
* the published train/test split.

Each split runs twice: with the measured water pKa as the anchor, and
with an in-fold predicted water pKa. Calibration: in each
leave-one-solvent-out fold, k = 0, 1, 2, 5, 10 measured pKas from the
held-out solvent shift each model's predictions, and the rest of that
solvent is scored (20 random draws).

Solvent descriptors (`umapka/solvent_descriptors.py`): 1/ε, Kamlet–Taft
α, β, π* and E_T(30).

**Status:** the code has been run end to end on synthetic data only, as
a correctness check. The real Nevolianis data is on Zenodo, which was
not reachable from the development environment. Run on Colab; the
script downloads the Zenodo record itself, once, into `./anion_data`:

```python
%cd /content/umapka
!git pull
# RDKit site features (a few minutes)
!python dev/solvent_shift_experiment.py zenodo --features rdkit
# site-focused UMA features (GPU; cached on Drive, resumable)
!python dev/solvent_shift_experiment.py zenodo --features uma \
    --cache /content/drive/MyDrive/umapka/solvent_shift_uma_cache.pkl
```
Outputs: `results/solvent_shift_{rdkit,uma}/summary.txt`, `per_fold.csv`,
`calibration.csv`, `calibration_{measured,predicted}.png`.

**How to read it.** The hypothesis is supported only if M4 beats M2
under leave-one-solvent-out, at k = 0 and at small k, for both water
anchors. If M2 with one calibration point matches M4, then UMA is not
what makes new solvents work; the calibration point is.

### Results (Colab, real data; `results/solvent_shift_{rdkit,uma}/summary.txt`)

Data: 8,220 of 8,223 D2A-pKa rows; 6,481 reactions. The non-water rows
are 60% DMSO and 13% MeCN; NMP (28) and ethylene glycol (22) are tiny.
1,291 non-water rows have a measured water pKa for the same reaction.

**1. The hypothesis, as stated, is not supported.** Under
leave-one-solvent-out with no calibration (k = 0), no model predicts an
unseen solvent usefully (MAE 2.8–4.9), and M4 does not beat M2 (UMA,
measured anchor: 2.99 vs 3.06; predicted anchor: 4.02 vs 3.95). The
shift formulation brings no advantage for unseen solvents.
Acetonitrile is unpredictable by every model (7.5–12 units): its
proton-solvation behaviour lies outside the other seven solvents.

**2. Continuous solvent descriptors do help the absolute model.**
M1 → M1b under LOSO: 3.55 → 2.77 (UMA, measured) and 4.10 → 3.38
(predicted). DMF drops from 8.8 to 0.9 because the descriptors place it
next to DMSO.

**3. One calibration measurement is the dominant effect.** Mean over
held-out solvents, UMA features:

| k measured pKas in the new solvent | 0 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|
| M1b, water pKa unknown (realistic) | 3.22 | 1.86 | 1.57 | 1.44 | 1.31 |
| M1b, water pKa measured | 3.20 | 1.23 | 1.02 | 0.92 | 0.81 |
| M2 offset only, water pKa measured | 3.36 | 1.82 | 1.65 | 1.47 | 1.36 |

A single point removes about 40–60% of the error. After calibration,
the learned models beat the pure offset by about 0.5–0.6
(1.23 vs 1.82 at k = 1), so the solute-specific response to the
solvent is real and learnable.

**4. What UMA adds, compared with RDKit site features on the same splits:**
* Much better absolute and aqueous chemistry. Random split, M1: 1.01 →
  0.72. Calibrated M1b at k = 10: 1.07 → 0.81 (measured anchor) and
  1.63 → 1.31 (predicted).
* **No gain for the shift given a measured water pKa**: M4 is identical
  (k = 1: 1.27 vs 1.28; k = 10: 0.90 vs 0.90).

UMA therefore improves the intrinsic chemistry, not the solvent
response.

**5. Anchoring to measured water pKa is what generalizes across
chemical families.** Leave-one-family-out, measured anchor: M2 offset
1.67 vs absolute M1 2.41–2.70. The trivial model wins here.

**6. The published split is not comparable to the 0.58 of Nevolianis
et al.** Only non-water rows are scored (n = 239, or n = 112 with a
measured water pKa). UMA M1b scores 0.70 there.

**Revised claim, supported by these data:** a model cannot yet predict
pKa in a solvent it has never seen, but with continuous descriptors,
site-focused UMA features and **one** measured pKa in the new solvent,
error falls from about 3.2 to 1.9 (1.2 if the water pKa is known), and
to 1.3 (0.8) with ten measurements.

**Next:**
1. Per-solvent calibration curves (in `calibration.csv`), so that MeCN
   does not dominate the mean.
2. Repeat with several seeds.
3. Add a proton-solvation descriptor, such as the solvent's
   autoprotolysis constant or a transfer free energy of H⁺ from the
   Zheng/Green data, the one physical term the current descriptors miss
   (MeCN).
4. A genuinely external solvent set.
