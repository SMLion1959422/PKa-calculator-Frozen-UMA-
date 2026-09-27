# pKa prediction: literature review and gap analysis for umapka

This review places umapka against the field, says plainly where the
original tool stood, and identifies the gaps it can credibly fill. It
motivates the methodology changes in [`METHODOLOGY.md`](METHODOLOGY.md).

> **Citation note.** Bibliographic details were checked against search
> indexes. Publisher sites were not reachable from the environment this
> was written in, so check volume/page numbers against the DOI before
> formal use.

---

## 1. The starting paper: Kido, Sato & Sakaki (2009)

K. Kido, H. Sato, S. Sakaki, *First principle theory for pKa prediction
at molecular level: pH effects based on explicit solvent model*,
**J. Phys. Chem. B 2009, 113, 10509–10514** (PubMed 19572664).

**What they did.** RISM-SCF-SEDD couples electronic-structure theory
for the solute with the reference interaction site model (RISM), a
statistical mechanics of molecular liquids. The acidic solution is
treated as a three-component liquid (water, a proton species, an
anion), and pH is set by controlling the proton-species number
density. Free energies of deprotonation of glycine's carboxyl and amino
groups were computed across pH, giving a titration curve and pKa "from
first principles" with the ionic environment explicit. The authors
describe it as the first attempt to evaluate pKa with ionic influence
taken into account by a statistical molecular theory. A follow-up
(Kido et al., *Int. J. Quantum Chem.* 2012) assessed pKa and pKb of amino
acids systematically with the same method.

**Why it matters for umapka.** It makes three conceptual points that
the original umapka ignored:

1. **pKa is a property of an equilibrium ensemble under stated
   conditions**, not of one molecule in isolation. The experimental
   observable is a titration curve.
2. **The ionic environment changes the answer.** Counter-ions and ionic
   strength shift apparent pKa and screen interactions.
3. **Glycine is the canonical test.** Its zwitterion is the dominant
   neutral species, and a method that treats the COOH and NH2 sites
   independently gets both pKas wrong by about 2 units. umapka's README
   listed exactly this ("zwitterionic carboxyls ... predicts ~5–8 where
   truth is ~2.2").

**Its limitations.** It is expensive (one molecule per calculation),
it was demonstrated on one small molecule, and first-principles
free-energy schemes of this type generally need an empirical
linear correction to reach chemical accuracy. The same group's later
3D-RISM-SCF work uses a "linear fitting correction scheme"
(Fujiki et al., *PCCP* 2018, 20, 27272). It is not a tool for
screening drug-like libraries.

---

## 2. Families of methods

### 2.1 Physics-based: thermodynamic cycles with solvation models

pKa = ΔG_deprot / (RT ln 10), with ΔG from a gas-phase calculation plus
solvation free energies from a continuum model (PCM, SMD, COSMO-RS).
Reviews: Alongi & Shields, *Annu. Rep. Comput. Chem.* 2010, 6, 113; Ho
& Coote, *Theor. Chem. Acc.* 2010, 125, 3; Fujiki et al., *J* 2021, 4,
849.

* **Direct/absolute schemes** fail by several units, mainly because
  of solvation-energy errors for ions. 1 pK unit is only 1.36 kcal/mol
  (59 meV), which is the point umapka's README makes.
* **Relative/isodesmic schemes and per-group linear calibration**
  cancel most of that error. Examples: COSMO-RS with a fitted slope
  (Klamt et al., *J. Phys. Chem. A* 2003, 107, 9380); semi-empirical
  QM with isodesmic references (Jensen, Swain & Olsen, *J. Phys. Chem.
  A* 2017, 121, 699), reaching roughly 1 pK for drug-like molecules;
  SMD/DFT in SAMPL6 (Zeng, Jones & Brooks, *JCAMD* 2018).
* **Cluster-continuum** (a few explicit waters around the ionized site)
  fixes the worst anion errors (Pliego & Riveros, *J. Phys. Chem. A*
  2001, 105, 7241).
* **Explicit / statistical-mechanical solvent** covers RISM and 3D-RISM
  (Kido 2009, Fujiki 2018) and constant-pH molecular dynamics (Mongan,
  Case & McCammon, *J. Comput. Chem.* 2004, 25, 2038; Swails, York &
  Roitberg, *JCTC* 2014, 10, 1341). These are the most rigorous
  treatments of pH and ions and the most expensive.

**Implication for umapka.** The README's claim that "computing
energies directly does not work" holds for *absolute* ΔG. It does not
hold for ΔG used *within a functional group* as a linear-free-energy
descriptor, which is how every successful QM pKa scheme uses it.
umapka computed UMA's energy on every forward pass and threw it away.

### 2.2 Empirical: Hammett–Taft and fragment rules

Perrin, Dempsey & Serjeant, *pKa Prediction for Organic Acids and
Bases* (Chapman & Hall, 1981): parent-group pKa plus substituent σ
constants with through-bond attenuation. This is the basis of the
commercial tools ACD/Percepta, ChemAxon Marvin and Schrödinger Epik
(Shelley et al., *JCAMD* 2007, 21, 681; Epik 7 is ML-based: Johnston
et al., *JCTC* 2023). These are still the practical baselines in
pharma. The same book gives the temperature rules used in §4.

### 2.3 Machine learning on experimental data

* Fingerprint/descriptor models: random forests on ECFP-type features.
  Baltruschat & Czodrowski, *F1000Research* 2020, 9, 113 released the
  open ChEMBL/DataWarrior training set plus the Novartis and
  AvLiLuMoVe external sets that umapka uses, with MAE about 0.68 in
  cross-validation.
* Graph neural networks: MolGpKa (Pan et al., *JCIM* 2021, 61, 3159,
  atom-centred, site-specific); pkasolver (Mayr et al., *Front. Chem.*
  2022, 10, 866585, transfer learning, sequential protonation states).
* Hybrid QM + ML: QupKake (Abarbanel & Hutchison, *JCTC* 2024, 20,
  6946) uses GFN2-xTB features in a GNN, with a QM tautomer search and
  site enumeration. It reports micro-pKa RMSE 0.5–0.8 on experimental
  test sets.
* **Thermodynamically consistent ML**: Uni-pKa (Luo et al., *JACS Au*
  2024) predicts per-*microstate* free energies and derives macro-pKa
  from the microstate ensemble. It reports Novartis/SAMPL6/SAMPL7 MAE
  0.81/0.49/0.55 and 0.62 on blind SAMPL8. Starling (Rowan, ChemRxiv
  2025, doi 10.26434/chemrxiv-2025-t8s9z) is a lighter Uni-pKa-style
  model with conformer-aggregated microstate energies and applications
  to pI, logD and BBB permeability.

**Trend.** Since about 2023 the field has moved from "one number per
molecule" to **microstate ensembles with thermodynamic consistency**,
the ML version of Kido's point 1.

### 2.4 Machine-learned interatomic potentials

Foundation MLIPs trained on DFT with charge and spin: UMA (Wood et al.,
arXiv 2506.23971, trained on OMol25, Levine et al., arXiv 2505.08762)
and AIMNet2 (Anstine, Zubatyuk & Isayev, *Chem. Sci.* 2025, 16, 10228).
They give DFT-quality energies at force-field cost, but not solvation
or free energies. Starling pairs AIMNet2 with implicit solvation for
related properties. **Using a frozen MLIP's internal representation
as a pKa featurizer, which is umapka's premise, is not established in
the literature.** That is a genuine novelty, but it has to beat simpler
baselines to matter (see §3).

### 2.5 Non-aqueous solvents

Data are scarce. Sources include the iBonD/Bordwell DMSO compilations
used in "holistic" multi-solvent ML (Yang et al., *Angew. Chem. Int.
Ed.* 2020, 59, 19282), the Tartu acetonitrile and DMSO scales, and the
curated 8-solvent set of Nevolianis et al. (ChemRxiv 2025, doi
10.26434/chemrxiv-2025-8bj2t; Zenodo 15604045) that umapka's
multisolvent model trains on. A 2025 review (Zheng et al., *J. Comput.
Chem.* 2025, doi 10.1002/jcc.27517) covers the field. It is much less
crowded than aqueous prediction.

### 2.6 Benchmarks

SAMPL6 (Işık et al., *JCAMD* 2018, 32, 1117), SAMPL7 (Bergazin et al.,
*JCAMD* 2021, 35, 771) and SAMPL8 are blind, macro-pKa, and multiprotic
on purpose. Novartis and AvLiLuMoVe are the standard external sets for
ML. Experimental pKas of the same compound from different labs differ
by several tenths of a unit (Settimo, Bellman & Knegtel, *Pharm. Res.*
2014, 31, 1082), which sets a floor on achievable MAE.

---

## 3. Where umapka stood (honest assessment)

| | umapka (as found) | Reference points |
|---|---|---|
| Novartis MAE | 1.17 | ECFP4 RF 1.45 (README); Uni-pKa 0.81 |
| AvLiLuMoVe MAE | 0.70 | |
| Multiprotic / zwitterions | Wrong by construction (independent sites) | Uni-pKa, Starling, Epik: microstate ensembles |
| pH / T / ionic strength | Post-hoc Davies shift on one site only | Kido 2009: explicit; Uni-pKa: pH-dependent populations |
| Site assignment | First SMARTS match | MolGpKa, QupKake: learned site enumeration |

The critique that the tool "doesn't address a gap" was fair.

* **It was not competitive** on the standard benchmark: 1.17 vs 0.81.
* **Its main failure modes were known and already solved elsewhere**:
  zwitterions, polyprotic molecules and macro-pKa.
* **Two avoidable errors, diagnosed in this revision**, account for much
  of the gap (`results/site_ablation.txt`):
  1. *Site mis-assignment.* The SMARTS priority rule picks a different
     atom from the dataset's site annotation, or no site at all, for
     **23% of Novartis** molecules (16.8% wrong site, 4.3% site not
     covered, 1.8% none) and 15% of the training set. On AvLiLuMoVe it
     agrees 100%, which matches that set's much lower error. Tetrazoles,
     a common carboxylic-acid isostere, could never be deprotonated
     because the SMARTS pointed at the ring carbon.
  2. *Global pooling.* RESULTS.md shows error growing with molecular
     size, the signature of diluting a local property over the whole
     molecule.

  With a plain RDKit featurizer, just correcting these two things
  reaches Novartis 0.99 with the annotated site and **1.08 fully
  automatic**, versus 1.48 for global features. That already beats the
  UMA model as shipped.

---

## 4. Gaps umapka can credibly fill

Ranked by how defensible the claim is:

1. **Condition-aware, thermodynamically consistent pKa from an open,
   inspectable model.** Most ML predictors return a single number at
   unspecified conditions (implicitly about 25 °C and low ionic
   strength). Pharmacology happens at **37 °C and I ≈ 0.15 M**. Amine
   pKas fall by about 0.3 units from 25 to 37 °C, and ionic strength
   shifts both pKas and site-site coupling. Kido 2009 showed why the
   ionic environment should be explicit. The new microstate layer does
   this with textbook physics: screened Coulomb coupling, Davies
   activities, and van 't Hoff group rules. It has **one fitted
   constant, validated with no dependence on the ML model**: LOO MAE
   0.32 on pKa₂−pKa₁ of 17 symmetric diacids/diamines, vs 1.01 for
   independent sites. It reproduces glycine's zwitterion and pI (6.06
   vs 6.06 experimental). Commercial tools model some of this, but
   open ML tools mostly do not.

2. **Non-aqueous and mixed solvents.** This area is under-served. The
   multisolvent model (8 solvents) plus the Yasuda–Shedlovsky mixture
   module is a real niche, provided its extrapolation limits stay
   documented (it fails leave-one-solvent-out for aprotic solvents).

3. **Testing whether MLIP foundation-model representations help pKa.**
   This is novel but not yet demonstrated. It needs the fair comparison
   in `dev/train_site_model.py` (same sites, same site-local pooling,
   UMA vs RDKit features). A negative result would still be publishable
   and useful.

4. **Sequential/polyprotic prediction with physical coupling.** Uni-pKa
   and Starling solve this with learned microstate energies. umapka's
   version is cheaper and interpretable: learned *intrinsic* site pKas
   plus physical coupling, the protein-electrostatics decomposition of
   Tanford & Kirkwood (*JACS* 1957, 79, 5333), Bashford & Karplus
   (*Biochemistry* 1990, 29, 10219) and Ullmann (*J. Phys. Chem. B*
   2003, 107, 1263). The decomposition is testable piece by piece, as
   done here for the coupling.

**Gaps it does not fill, and should not claim:** best-in-class
aqueous accuracy (Uni-pKa, Epik and QupKake are ahead); tautomer
enumeration; protein pKa.

---

## 5. Open problems the literature agrees on

* Site and tautomer ambiguity in training labels (addressed here in
  part through annotated-site training).
* Very low (<2) and high (>12) pKa, where data are sparse.
* Uncertainty estimates and applicability domain.
* Multiprotic molecules with coupled tautomers (fused heteroaromatics).
* Transfer to solvents never seen in training.
* Experimental noise at the 0.3–0.5 unit level (Settimo 2014), which
  caps any benchmark.

---

## References (grouped as cited)

* Kido, Sato, Sakaki. J. Phys. Chem. B 2009, 113, 10509. · Kido et al. Int. J. Quantum Chem. 2012.
* Fujiki, Matsui, Shigeta, Nakano, Yoshida. J 2021, 4, 849 (review). · Fujiki et al. PCCP 2018, 20, 27272.
* Alongi, Shields. Annu. Rep. Comput. Chem. 2010, 6, 113. · Ho, Coote. Theor. Chem. Acc. 2010, 125, 3.
* Klamt et al. J. Phys. Chem. A 2003, 107, 9380. · Jensen, Swain, Olsen. J. Phys. Chem. A 2017, 121, 699.
* Pliego, Riveros. J. Phys. Chem. A 2001, 105, 7241. · Zeng, Jones, Brooks. JCAMD 2018.
* Mongan, Case, McCammon. J. Comput. Chem. 2004, 25, 2038. · Swails, York, Roitberg. JCTC 2014, 10, 1341.
* Perrin, Dempsey, Serjeant. pKa Prediction for Organic Acids and Bases. 1981.
* Shelley et al. JCAMD 2007, 21, 681. · Johnston et al. JCTC 2023 (Epik 7).
* Baltruschat, Czodrowski. F1000Research 2020, 9, 113.
* Pan et al. JCIM 2021, 61, 3159 (MolGpKa). · Mayr et al. Front. Chem. 2022, 10, 866585 (pkasolver).
* Abarbanel, Hutchison. JCTC 2024, 20, 6946 (QupKake).
* Luo et al. JACS Au 2024 (Uni-pKa). · Rowan, ChemRxiv 2025, 10.26434/chemrxiv-2025-t8s9z (Starling).
* Wood et al. arXiv 2506.23971 (UMA). · Levine et al. arXiv 2505.08762 (OMol25). · Anstine, Zubatyuk, Isayev. Chem. Sci. 2025, 16, 10228 (AIMNet2).
* Yang et al. Angew. Chem. Int. Ed. 2020, 59, 19282. · Nevolianis et al. ChemRxiv 2025, 10.26434/chemrxiv-2025-8bj2t. · Zheng et al. J. Comput. Chem. 2025, 10.1002/jcc.27517.
* Işık et al. JCAMD 2018, 32, 1117 (SAMPL6). · Bergazin et al. JCAMD 2021, 35, 771 (SAMPL7).
* Settimo, Bellman, Knegtel. Pharm. Res. 2014, 31, 1082.
* Tanford, Kirkwood. JACS 1957, 79, 5333. · Kirkwood, Westheimer. J. Chem. Phys. 1938, 6, 506. · Bashford, Karplus. Biochemistry 1990, 29, 10219. · Ullmann. J. Phys. Chem. B 2003, 107, 1263.
* Malmberg, Maryott. J. Res. Natl. Bur. Stand. 1956, 56, 1. · Davies. Ion Association, 1962.
