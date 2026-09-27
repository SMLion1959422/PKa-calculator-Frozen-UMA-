# umapka methodology v3

What changed, why, and what has and has not been validated. Background
and citations are in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).

## Pipeline

```
SMILES
  │
  ├─ 1. site enumeration ─────── umapka/sites.py (SMARTS + charge-sharing atoms)
  │
  ├─ 2. intrinsic micro-pKa ──── PkaPredictor.predict_site, one per site, others neutral
  │       site_v3 models: UMA embeddings pooled AROUND the site + UMA ΔE
  │       pair_v1 models: whole-molecule pooling (original)
  │
  ├─ 3. microstate ensemble ──── umapka/microstates.py
  │       all 2^N protonation states; screened site-site Coulomb coupling;
  │       Davies activity (ionic strength); van 't Hoff group rules (T)
  │
  └─ 4. observables ──────────── macro pKas, per-site apparent pKas, pI,
                                 net charge / species / uncharged fraction vs pH
```

`PkaPredictor.predict_macro()` and `predict_pka.py --macro` run the
whole chain. `predict()` / `predict_site()` still return the single
intrinsic value, as before.

## 1. Thermodynamic layer (validated)

For microstate *x* (x_i = 1 if site *i* holds its proton):

```
G(x)/(RT ln10) = Σ_i x_i (pH − pKa_i)  +  Σ_{i<j} W_ij q_i q_j  −  log10 γ(z_x)
```

The intrinsic pKa_i is the "other sites neutral" micro-constant, which
is exactly what the regressor predicts. The site charge q is 0/−1 for
acids and +1/0 for bases. Macro pKas come from the binding polynomial,
pK_n = log10(Z_n/Z_{n−1}) (Ullmann 2003).

**Coupling.** W_ij = ⟨Σ_ab w_a w_b · 332·e^{−κr}/(ε(r)·r)⟩ / (RT ln10),
with ε(r) = min(D·r, ε_solvent), averaged over 20 ETKDG/MMFF conformers.
Carboxylate charge is split over both oxygens, guanidinium over three
nitrogens, and so on.

**Validation, independent of any ML model.** For a symmetric two-site
molecule, pKa₂ − pKa₁ = log10 4 + W, so the intrinsic pKa cancels.
Fitting D on 17 diacids/diamines (`data/symmetric_polyprotic.csv`)
gives D = 9.6 (ε ≈ 48 at 5 Å, the bulk-water value by about 8 Å):

| Model for pKa₂ − pKa₁ | MAE (pK units) |
|---|---|
| independent sites (statistical factor only) | 1.01 |
| constant gap | 0.80 |
| **microstate model, leave-one-out** | **0.32** |

The largest residuals are piperazine (−0.8 in-sample; it is a rigid ring
with the charges 2.9 Å apart) and succinic acid (+0.6). Several
literature values differ between compilations by 0.1–0.3, which is
noted in the CSV.

**Held-out zwitterion (glycine).** Given the intrinsic micro-pKas
implied by the experimental microconstants (4.38 COOH, 7.75 NH₃⁺), the
model gives macro pKas 1.81 / 10.32 (exp 2.35 / 9.78; independent sites
4.38 / 7.75), pI 6.06 (exp 6.06), and 99.95% zwitterion at pH 7. The
NH₃⁺/COO⁻ attraction is overestimated by about 0.5 pK. A likely cause
is that the neutral molecule's conformer ensemble is used for every
microstate.

Reproduce: `python dev/validate_coupling.py` → `results/coupling_validation.txt`.

**Conditions.**
* *Ionic strength*: the Davies γ of each microstate's net charge, plus
  Debye screening e^{−κr} of W. pH is then −log a_H⁺ (electrode pH).
  This is not yet validated against measured data; see §4.
* *Temperature*: ε_water(T) (Malmberg–Maryott), RT scaling of W, and
  group rules for the intrinsic pKa. For bases this is Perrin's
  −dpKa/dT = (pKa − 0.9)/T, which reproduces methylammonium's ΔH of
  55 kJ/mol. Carboxylic-type acids get ≈ 0, phenol-type O/S/N–H acids
  −0.012 /K.
* *Solvent*: ε cap and Davies constant use the solvent's permittivity.
  D was fitted in water only, so treat non-aqueous coupling as
  qualitative.

## 2. Site assignment (diagnosed; training fix provided)

The original pipeline chose the site by SMARTS priority (acids first,
then first match). Against the site annotations shipped with the
Baltruschat & Czodrowski data (`marvin_atom`):

| Set | SMARTS picks annotated atom | wrong site | site not covered | no site |
|---|---|---|---|---|
| training (5994) | 85.1% | 4.1% | 3.2% | 7.7% |
| Novartis (280) | 77.1% | **16.8%** | 4.3% | 1.8% |
| AvLiLuMoVe (123) | 100% | 0 | 0 | 0 |

That explains much of the Novartis vs AvLiLuMoVe gap (1.17 vs 0.70).
Most wrong picks are "which pyridine-type N" in polyazines. This
revision also fixes a bug: both tetrazole SMARTS used the ring carbon
as the site atom, so tetrazoles were never deprotonated.

`dev/train_site_model.py` builds each training pair at the annotated
atom, so labels match features. At inference, site choice is still by
priority. A learned site-ranker and "most acidic/basic" rules were
tried and did not beat it (Novartis 1.09 and 1.35 vs 1.08; see
`results/site_ablation.txt`), so choosing the site automatically
remains open.

## 3. Site-local representation (ablated with RDKit features; UMA run pending)

`pool_site` builds each charge state's representation from four
128-dim blocks: the mean over the site's charge atoms, Gaussian-weighted
means at σ = 2.5 Å and 5 Å from the site, and the global mean. The pair
feature is [h_p ; h_d ; h_p − h_d ; ΔE_UMA ; n_heavy], 1538 dimensions.
ΔE is UMA's deprotonation energy from the same forward passes; §2.1 of
the review explains why it is useful as a within-group descriptor even
though it is useless as an absolute pKa.

Controlled RDKit-only ablation (`dev/site_ablation.py`, same LightGBM,
same splits):

| Arm | Novartis | AvLiLuMoVe |
|---|---|---|
| A: global Morgan counts | 1.48 | 0.77 |
| B: + site-centred features at the SMARTS site | 1.15 | 0.59 |
| **B': trained at annotated sites, SMARTS site at test (fully automatic)** | **1.08** | **0.53** |
| C: + site-centred features at the annotated site | 0.99 | 0.53 |
| *reference: shipped UMA model_core_v2* | *1.17* | *0.70* |
| **UMA site_v3, SMARTS site at test (automatic)** | **1.03** | **0.43** |
| **UMA site_v3, annotated site** | **0.86** | **0.43** |
| *reference: Uni-pKa (published)* | *0.81* | – |

**UMA result (Colab T4, single run, scaffold CV 0.72):** the rows above in bold. UMA beats the RDKit arms on every split, so the representation adds information once pooled around the site. Originally pending: It needs a
GPU and access to the gated `facebook/UMA` weights, neither of which was
available in the environment this was developed in. Run:

```bash
git clone https://github.com/czodrowskilab/Machine-learning-meets-pKa mlpka
python dev/train_site_model.py mlpka/datasets   # -> models/model_site_v3.pkl
```

It prints scaffold-CV and external MAE for both site modes. **Compare
against arm B′ above.** If UMA does not beat 1.08 / 0.53, the honest
conclusion is that the RDKit site-local model should be the default
featurizer, with UMA kept for ΔE and conformational features only.

## 4. Benchmark on measured multi-pKa molecules

`dev/benchmark_multiprotic.py` tests the thermodynamic layer against
experiment. Both arms get the **same** intrinsic per-site pKas, so only
the layer differs:

* **independent**: sorted site pKas (the pre-v3 `predict_all_sites`),
  with charge and pI from independent sites;
* **microstate**: coupled macro pKas, with the measurement's ionic
  strength applied.

Intrinsic pKas come from the GPU-free `umapka.rdkit_site` model,
retrained with every benchmark molecule removed. Predicted pKas are
matched to experimental ones by Hungarian assignment, as in the SAMPL6
analysis. Output is in `results/benchmark_multiprotic.{txt,csv}`.

| Set (n pKas) | independent MAE | microstate MAE | per-molecule change [95% CI] | notes |
|---|---|---|---|---|
| amino acids + polyprotic (60) | 1.40 | **0.84** | **−0.55 [−0.73, −0.38]**, better 24/26, p = 2e-6 | pI MAE 0.57 → 0.38; charge at pH 7.4 correct 25 → 26/26 |
| SAMPL6 (31) | 1.52 | 1.51 | +0.04 [−0.27, +0.36], n.s. | spurious in-window pKas 26 → 12 |
| SAMPL7 (20, monoprotic) | 1.99 | 2.05 | +0.06 [+0.02, +0.10], worse 16/20 | entirely from the I = 0.15 term (1.99 → 2.00 at I = 0) |

What this shows:

* **Where sites interact, the layer works.** Zwitterions and polyprotic
  acids improve substantially and significantly, with the correct
  charge state for every molecule. A residual +0.26 bias matches the
  ~0.5 over-coupling seen for glycine in §1.
* **On drug-like SAMPL molecules it is neutral for pKa values** but
  removes about half of the spurious pKas predicted in the 2–12 window.
  That matters for charge-state and logD work.
* **The ionic-strength term is not validated by these data.** Its
  direction is physically right: the SAMPL values are apparent pKas in
  0.15 M KCl (documented for SAMPL6, assumed for SAMPL7), where acids
  appear ~0.1 more acidic. But the intrinsic model is trained on ChEMBL
  labels measured at unknown, often similar, ionic strength, so applying
  the correction on top can double-count. Leave `salt=` unset unless
  your intrinsic pKas are known to refer to I ≈ 0.
* **The bottleneck is per-site accuracy on unfamiliar chemotypes.**
  SAMPL7 has a −1.6 bias (sulfonamides predicted too acidic). For
  comparison, Uni-pKa reports SAMPL6/SAMPL7 MAE 0.49/0.55. This is the
  part a better representation, possibly UMA, has to fix. To test it:
  `python dev/benchmark_multiprotic.py <datasets> --uma models/model_site_v3.pkl`.

## 5. Remaining limitations

* Tautomers beyond proton moves between detected sites are not
  enumerated.
* One (neutral) conformer ensemble serves all microstates, so salt
  bridges are under-weighted.
* The coupling constant was fitted on 17 molecules. A larger polyprotic
  set, for example the IUPAC digitized aqueous constants (non-commercial
  licence), should be used to refit and test it on asymmetric molecules.
* No uncertainty estimate yet. The natural next step is split-conformal
  intervals calibrated per site group on scaffold-CV residuals.
* Intrinsic pKas outside 2–12 are still extrapolations of the regressor.

## Tests

`python -m pytest` (no GPU needed; UMA is stubbed) covers:
single-site identity, the statistical factor, path independence of the
macro-pKa sum, the glycine zwitterion, monotone titration, screening
and activity signs, temperature rules, Debye length, charge sharing,
the tetrazole site, order-preserving ionization, locality of
`pool_site`, the `predict_macro` wiring, and the `rdkit_site` save/load path.
