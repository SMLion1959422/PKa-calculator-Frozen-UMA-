# Repo map

159 Python files at the root, most of them one-off experiments. This is
what's actually load-bearing, so you don't have to guess when you come
back to push MAE below 0.918.

## The shipped pipeline

| file | role |
|---|---|
| `predict_pka.py` | CLI. One molecule, any solvent/salt/mixture. |
| `pka_server.py` | **Use this for more than one molecule.** Loads once (~63 s) then ~0.7 s each, with a persistent cache. A fresh `predict_pka.py` process re-reads the 1.17 GB UMA checkpoint every time. |
| `predict_microstates.py` | Polyprotic / zwitterions. Full 2^n microstate ensemble. |
| `predict_ladder.py` | Greedy sequential ladder; fallback when microstates hits its site cap. |
| `umapka/` | The package. `predictor.py` (UMA + pooling), `site_finder.py` (ranker + kind + ballot expansion), `electronic.py` (descriptors + scoring), `microstates.py`, `solvents.py`, `solvation.py`, `mixtures.py`. |
| `test_all.py` | 22 end-to-end tests. Run before trusting anything. |

## Models actually loaded at inference

Everything else in `models/` is a documented dead end and is gitignored.

| file | what it does |
|---|---|
| `model_core_v20_ensemble.pkl` | aqueous regressor (default) - 3-seed bagged LGBM + ridge, isotonic |
| `model_core_v3.pkl` | base regressor; also the embedding stack loader |
| `model_core_v16_elec.pkl` | previous hybrid, kept as fallback |
| `site_finder_v2.pkl` | LambdaRank site ranker (97.4% atom accuracy) |
| `kind_classifier.pkl` | acid vs base (100% on Novartis, on the correct atom) |
| `site_detector.pkl` | hard-negative ballot expansion (reaches sulfonamides, thiols, phosphonic acids) |
| `multisolvent_tuned.pkl` | non-aqueous solvents |

## Measuring anything

| file | role |
|---|---|
| `cache_external_features.py` | one UMA pass over the held-out sets -> `feat_external_learned.pkl` |
| `eval_cached.py` | **scores models in seconds** from that cache. Use this; do not re-embed per model. |
| `eval_union_hybrid.py` | site accuracy, baseline vs ballot expansion |
| `check_test_novelty.py` | Tanimoto novelty of a test set vs training - run this before believing any gain |
| `validate_multisolvent.py` | non-aqueous, production path |
| `build_polyprotic_benchmark.py` | polyprotic (label-noisy - read the caveats) |

## Training

`train_v20_ensemble.py` (current default), `train_hard_negatives.py`,
`train_kind_classifier.py`, `train_site_finder_v2.py`,
`embed_core_v6.py` (rebuilds the big feature cache).

`train_v21_maxdata.py` and `train_v22_sizeweighted.py` are kept
deliberately: both are **negative results** (see RESULTS.md section 6),
and keeping them stops the same ideas being retried.

## Regenerable, not in git

`feat_*.pkl`, `core_maxdata.pkl` - the two largest exceed GitHub's
100 MB limit. Rebuild with `embed_core_v6.py` /
`cache_external_features.py`. `feat_external_learned.pkl` is the one
worth keeping locally; everything downstream of it scores in seconds.

## Before the next experiment

Read RESULTS.md section 6 first. Eleven approaches have been measured and
ruled out, several of which look obviously correct beforehand (L1 loss
when optimising MAE; higher-L spherical-harmonic channels; SASA). The
pattern across the whole session:

**Every gain came from site assignment. Nothing from the regressor.**

    site assignment : 1.001 -> 0.918
    regressor       : 11 approaches, all null or negative

Site assignment is now ~solved (98.6% kind agreement), so the remaining
0.077 to the 0.841 wall is site errors, and everything past that needs
new data - specifically molecules scaffold-matched to Novartis, not just
size-matched. v21 proved size-matching alone makes things worse.

Two conventions worth not breaking:
- select on OOF, score Novartis **once**. Five OOF gains here failed to
  transfer; Novartis is 275 molecules and tuning against it turns it into
  a validation set.
- report a paired bootstrap CI, not just a delta. Two "improvements"
  this session were inside the noise band.
