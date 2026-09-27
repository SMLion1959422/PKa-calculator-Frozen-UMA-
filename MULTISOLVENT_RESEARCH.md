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
