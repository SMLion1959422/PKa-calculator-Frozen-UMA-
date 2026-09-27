# Data

| File | What | Source / licence |
|---|---|---|
| `symmetric_polyprotic.csv` | pKa1/pKa2 of 17 symmetric diacids and diamines; **fit set** for the coupling constant in `umapka/microstates.py` | CRC Handbook / Perrin compilations, 25 °C; entries where compilations disagree are noted |
| `benchmark_multiprotic.csv` | 26 molecules with 2–3 macroscopic pKas: the 20 proteinogenic amino acids (with pI), ω-amino acids of increasing chain length, 4-aminobenzoic, citric and tartaric acid. **Not used in any fit.** | Amino acids: Lehninger *Principles of Biochemistry*, Table 3-1 (each listed pI equals the mean of its flanking pKas); others: CRC Handbook, 25 °C |
| `sampl6_pka.csv` | SAMPL6 blind-challenge macroscopic pKas (24 molecules, 31 pKas, measured range 2–12, 0.15 M KCl, 25 °C) | [samplchallenges/SAMPL6](https://github.com/samplchallenges/SAMPL6), MIT. Işık et al., *JCAMD* 2018, 32, 1117 and *JCAMD* 2021, 35, 131 |
| `sampl7_pka.csv` | SAMPL7 blind-challenge pKas (22 sulfonamide-type acids; two reported as ">12" are excluded) | [samplchallenges/SAMPL7](https://github.com/samplchallenges/SAMPL7), MIT code / CC BY 4.0 data. Bergazin et al., *JCAMD* 2021, 35, 771 |

`charge_7_4` in `benchmark_multiprotic.csv` is the dominant net charge
at pH 7.4 implied by the listed pKas and the known site types.

SAMPL files were extracted verbatim from the challenge repositories
(`pKa_experimental_values.csv`; `Experimental_Properties_of_SAMPL7_Compounds.csv`),
with counter-ions stripped from the SMILES.
