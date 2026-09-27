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

---

## 7. Is the calibration offset the proton transfer term?

The §6 result -- no model predicts an unseen solvent, but one measured
pKa removes 40-60% of the error -- has a candidate mechanism. Relative
to water,

    pKa(S) - pKa(W) = [dG_tr(A-) - dG_tr(HA)]/(RT ln10) + dG_tr(H+)/(RT ln10)

where the last term is one constant per solvent. Zheng/Green (J. Comput.
Chem. 2025) treat exactly this term as a per-solvent regression
parameter. If the offset we fit from k measured points IS that term,
the empirical calibration curve becomes a mechanistic claim, and
acetonitrile's failure is explained in the same breath: its dG_tr(H+)
is about +10.7 kcal/mol (~ +7.8 pK units), a large outlier, against
-4.6 for DMSO and +2.1 for methanol.

`dev/proton_offset_test.py` runs two tests:

* **Test 1 (internal, no external data).** Fit the offset on one
  chemical family, apply it to a different family in the same solvent.
  A solvent constant transfers; a fitted fudge factor does not. Scored
  against the family's own offset (cheating lower bound) and against
  predicting no shift.
* **Test 2 (external).** Regress the fitted offsets on literature
  dG_tr(H+) (`umapka/proton_transfer.py`). Slope ~1, intercept ~0
  supports the identification; reported with and without acetonitrile
  and over the verified-value subset.

**Validation.** On synthetic data with a KNOWN planted proton constant
plus a solute-dependent transfer term, the test recovers slope 1.01,
r^2 0.96 on the verified subset, and correctly reports the leftover mean
solute term as a nonzero intercept (+3.7). Test 1 correctly flagged the
family-dependent component that was planted. The diagnostic behaves as
intended; it has NOT yet been run on the real data.

**Caveats to state in any write-up.**
* The offset absorbs everything constant per solvent, including the mean
  solute transfer term over this dataset's chemistry. Agreement is
  evidence, not proof -- the synthetic test shows exactly this as a
  nonzero intercept with a correct slope.
* At most 7 solvents, so the correlation is small-n; NMP (28 rows) and
  ethylene glycol (22 rows) are thin.
* Single-ion transfer energies need an extrathermodynamic assumption and
  compilations disagree. Only Water/MeOH/MeCN/DMSO entries in
  `umapka/proton_transfer.py` are cross-checked; the rest are flagged
  UNVERIFIED and must be replaced from a primary compilation (e.g.
  Kalidas, Hefter & Marcus, Chem. Rev. 2000, 100, 819; or the 2026
  reassessment, Stroh et al., ChemPhysChem, doi 10.1002/cphc.202500349)
  before publication.

Run:  `python dev/proton_offset_test.py zenodo`

### Result (real data): the offset is the proton term PLUS anion desolvation

`results/proton_offset/summary.txt`. 1291 paired water/solvent rows.

**Test 1 passes.** An offset fitted on other chemical families and applied
to a held-out family gives MAE 1.65, against 0.97 for the family's own
offset and 7.12 for no shift: it captures 89% of the achievable gain
without seeing the target chemistry. The offset is largely a solvent
property.

**Test 2, as posed, fails.** Regressing offsets on literature dG_tr(H+)
gives r^2 0.60 over 6 solvents but **r^2 0.09 with acetonitrile removed**.
The apparent correlation was one leverage point. Every fitted offset is
larger than the proton term, by +3.4 to +10.7 pK units.

**What the residual is.** It splits by the solvent's H-bond DONOR ability:
+3.6 for protic (MeOH, EtOH), +9.2 for aprotic (MeCN, DMSO, DMF, NMP).
Regressed on d_alpha = alpha(S) - alpha(water), n = 5:

    offset = 0.99 * dG_tr(H+)[pK]  -  6.87 * d_alpha  +  1.96
    r^2 0.979 on the residual; leave-one-solvent-out MAE 1.14

Two things make this more than a curve fit, with one important
qualification on the first:
* the coefficient on the proton term is **0.99** from a JOINT fit (not
  pinned by construction), which is consistent with the 1.00 the
  thermodynamic cycle requires. But with n = 5 and 3 parameters the 95%
  CI is **[0.66, 1.32]**, which does not exclude much; freeing the
  coefficient cuts RSS only from 0.920 to 0.913. Consistent with the
  mandated value, NOT a confirmation of it. The separately quoted
  "r^2 0.979" comes from the sequential analysis, where that coefficient
  IS pinned to 1 - the two should not be cited side by side, and are now
  reported separately;
* the same residual shows **no** relation to beta, the H-bond ACCEPTOR
  scale (r^2 0.03). An anion accepts H-bonds, so the solvent's donor
  ability is the physically relevant axis and beta should be irrelevant.
  It is.

So the mechanism behind the one-point calibration is not the proton term
alone: it is proton transfer plus mean anion desolvation, and the second
piece is predictable from a tabulated solvent property.

**The experiment this sets up.** If the offset is predictable from
dG_tr(H+) and alpha with ~1.1 error, a new solvent might need *zero*
measurements rather than one. That is directly testable: substitute the
predicted offset into the k = 0 arm of the LOSO calibration and compare
against the measured k = 1 result (1.23 with a measured water pKa, 1.86
without). NOT YET RUN -- and note that 1.14 is the error on the offset,
while the pKa error also carries the per-molecule spread (IQR 0.6-3.0),
so the two are not directly comparable without running it.

**Limitations (all in the results file, and all publication-blocking
until addressed).** Five solvents and three parameters; restricted to
VERIFIED literature values it is 3 points and 3 parameters, i.e. zero
degrees of freedom, so the current testability rests on the DMF and
ethanol values flagged UNVERIFIED in `umapka/proton_transfer.py`.
Perturbing those by +-1 kcal/mol moves the alpha slope over
[-7.91, -5.93], so sign and scale survive but the coefficient does not.
The data cannot distinguish a continuous alpha dependence from a
two-state protic/aprotic one: LOO 0.60 vs 0.83 on five points is not a
distinguishable difference, and the only solvent making alpha continuous
is acetonitrile (alpha 0.19) - the same high-leverage point behind the
spurious r^2 0.60 in test 2. Say so rather than defend it; the two-term
decomposition survives either way. The three aprotics span 2.5 units with
no alpha variation at all, which may be a COMPOSITIONAL artifact rather
than chemistry (see below).

**Two prerequisites before anything is built on the offset**
(`dev/offset_confounds.py`):
* *Charge-type flip.* The leftover is claimed to be mean ANION
  desolvation, tracking the solvent's H-bond DONOR ability. A cationic
  acid (BH+ -> B + H+) puts the charge on the reactant as a cation, an
  H-bond DONOR stabilized by ACCEPTORS, so its leftover should track
  beta with a structurally different sign - a directional prediction no
  curve fit yields by accident, and it would double the offsets
  available for the same parameters. **But D2A-pKa is an anion solvation
  dataset written HA >> A-, and the family breakdown of its non-water
  rows contains no cationic-acid category at all.** Check A counts them;
  if it returns zero, this test needs iBonD or the Leito basicity
  scales (MeCN, THF), not this dataset.
* *Compositional confound.* The offset is a median over whatever
  chemistry each solvent happens to contain, and the set is 60% DMSO. If
  family composition differs across solvents, pooled offsets differ for
  non-solvent reasons. Check B recomputes them on a MATCHED set (only
  reactions measured in water and >= 2 non-water solvents). If the
  aprotic spread shrinks, the 2.5-unit anomaly was composition.

### Update: literature values re-sourced; DMF was wrong, and fixing it sharpens the result

TATB recommended values (kJ/mol): MeOH 8.7, EtOH 11.1, MeCN 44.8,
DMSO -19.4, DMF -14.4. Four reproduce the table's existing entries to
<= 0.05 kcal/mol. **DMF did not**: the unverified entry was -18.4 kJ/mol
against a sourced -14.4, an error of 0.96 kcal/mol (0.70 pK units).

| | before (bad DMF) | after |
|---|---|---|
| joint dG_tr(H+) coefficient | +0.990 | **+1.019** |
| its 95% CI | [0.658, 1.322] | **[0.841, 1.196]** |
| LOO MAE | 1.14 | **0.40** |
| aprotic spread | 2.06 | 1.36 |
| sequential r^2 vs alpha | 0.979 | 0.993 |
| specificity: r^2 vs beta | 0.032 | 0.041 |

The CI now **excludes 0.733** -- the coefficient that would be mandated
if the offset and the proton term were in mismatched units -- while
containing the 1.00 that matched units require. Before the correction
the interval contained both and discriminated nothing. The units
objection is now answered by the data rather than only by inspecting the
code.

About a third of the previously unexplained aprotic spread was a bad
literature input rather than chemistry. And no parameter was added and no
fit was freed to get there: one input was replaced with an independently
sourced value and the residuals fell, which is how a correct model
behaves given a corrected input.

Still outstanding: none of the five has been read off a PRIMARY table by
this project, and methanol alone is quoted at both 8.7 and 10.4 kJ/mol
(0.4 kcal spread) across compilations. NMP and ethylene glycol have no
located value and too few rows to use.

### Confound checks (real data): one test ruled out, the decomposition survives, and a replication appears

`results/offset_confounds/summary.txt`.

**Check A - definitive negative.** Zero of 6487 reactions are cationic
acids. D2A-pKa is `HA >> A-` by construction, so the
alpha(anionic) -> beta(cationic) sign-flip test cannot be run here at
all. It needs cationic-acid pKa in the same solvents (iBonD; the Leito
basicity scales in MeCN/THF).

**Check B - the offsets are not compositionally confounded.**
Recomputed on a matched set (reactions measured in water and >= 2
non-water solvents), every offset moves by <= 0.19 pK. The aprotic
residual spread shrinks only 1.36 -> 1.20. So the between-solvent
decomposition is not an artifact of family mix.

**But the pooled offset is a coarser object than it looked.** The
within-solvent spread of the median shift ACROSS families is 3-6 pK
(DMSO 2.42 to 8.72; MeCN 11.22 to 17.03), several times larger than the
1.2-1.4 pK between-solvent residual the decomposition explains. This is
the solute-dependent transfer term the cycle predicts, so it is expected
rather than anomalous - but it bounds what any single per-solvent
constant can ever deliver, and it should be stated plainly.

**The compensating find: the alpha dependence REPLICATES within each
family.** Regressing (family offset - proton term) on d_alpha
separately:

| family | slope | 95% CI | r^2 | rows |
|---|---|---|---|---|
| carboxylic | -6.92 | [-7.92, -5.92] | 0.994 | 655 |
| N-H | -6.41 | [-7.52, -5.29] | 0.991 | 172 |
| phenol | -6.03 | [-7.77, -4.29] | 0.976 | 239 |
| C-H | -2.84 | [-8.12, 2.44] | 0.728 | 74 |

Three chemically independent families give slope -6.0 to -6.9 with
r^2 > 0.97 on entirely different molecules. That is a real replication
and is stronger than the pooled 5-point fit, because it shows the
relationship is not an artifact of pooling heterogeneous chemistry.

**Tentative and underpowered.** The ordering carb > N-H > phenol > C-H
is the direction anion charge delocalization predicts (carboxylate
localized on two oxygens; phenolate delocalized into the ring; enolate-
type C-H anions most delocalized, so least H-bond dependent). But the
CIs overlap almost entirely and the C-H interval spans zero. Suggestive,
not established. Testing it properly needs more solvents per family or a
continuous delocalization descriptor rather than discrete families - and
that, not the charge-flip test, is now the most promising route to a
directional prediction using data already in hand.

### Continuous delocalization descriptor (`dev/delocalization_test.py`)

The family result (carb -6.92, N-H -6.41, phenol -6.03, C-H -2.84) is in
the direction anion charge delocalization predicts but has overlapping
CIs: six categories are too coarse. This replaces the category with a
continuous per-molecule descriptor.

**The UMA descriptor.** Deprotonation changes UMA's per-atom embeddings;
WHERE it changes them is where the charge went. Heavy-atom order is
preserved across the two species by `_ionize_keep_order`, so with
`w_a = ||h(A-)_a - h(HA)_a||` and `d_a` the bond distance from the site:

    R_deloc = sum_a d_a * w_a / sum_a w_a     mean response radius
    F_far   = fraction of the response > 2 bonds out
    IPR     = 1 / sum_a (w_a/sum w)^2         atoms sharing the response

Localized anion -> small values. These need no pKa data to compute, and
they are the natural thing a frozen embedding can supply that a
fingerprint cannot.

**Baselines it must beat**: the 6 discrete families; the identical
construction from Gasteiger charges (free, 1980 empirical model); and a
conjugated-atom count.

**Tests.** With `y = shift - dG_tr(H+)/RTln10` (the anion transfer term),
does the descriptor modulate the alpha slope?
`M0: y = a*d_alpha + b` against `M1: y = (a + c*D)*d_alpha + b + e*D`,
by nested F-test, plus leave-one-solvent-out and leave-one-family-out.

The decisive one is the **within-family** test: inside a single family
the label is constant, so a descriptor that merely encodes family
identity can do nothing, and a real effect should keep its sign across
families.

**Validation on synthetic data with planted per-family slopes**
(carb -7.0, phenol -5.0, C-H -2.0): the pooled interaction test fires
strongly (topo_conj F = 269 with the correct positive sign, meaning more
conjugated -> shallower slope), family dummies recover the planted
values (MAE 0.185 vs 0.810 for M0), and - the important part - the
**within-family test correctly finds NOTHING** (p = 0.3-0.5, signs
inconsistent), because constant-within-family sensitivity was what was
planted. It does not manufacture false positives.

Note on LOFO: with few families it is a severe extrapolation and every
descriptor did worse than M0 on the synthetic set. Beating M0 on LOFO
would be a strong result; failing it is close to uninformative when the
families are few. LOSO and within-family are the columns to read.

Run (UMA path needs a GPU; baselines run anywhere):

    python dev/delocalization_test.py zenodo --uma models/model_site_v3.pkl \
        --cache /content/drive/MyDrive/umapka/deloc_cache.jsonl \
        --out /content/out/delocalization

