# Merge notes: combining the aqueous and multisolvent branches

This repo merges `UMA_Pka-main` (aqueous-only, but with `predict_chain`
absent) and `UMA_Pka-multisolvent` (superset predictor + multisolvent
regressor) into one package, and adds molarity/salt handling to the CLI
plus a new mixed-solvent module. Nothing here required retraining
`model_core.pkl`, `model_core_v2.pkl`, or `multisolvent_tuned.pkl` — all
three are used as-is, exactly as they were trained.

## What changed

1. **One `PkaPredictor`, both branches' capabilities.** Started from
   the multisolvent branch's `predictor.py` (it's a strict superset:
   `predict_all_sites`, `predict_chain`, the free-energy ensemble path).
   `main`'s `predictor.py` had nothing this one doesn't.

2. **`solvent=` is now a real, working parameter on `predict()`,
   `predict_detailed()`, `predict_site()`, `predict_all_sites()`.**
   Before, solvent-awareness only existed as copy-pasted logic inside
   the standalone `predict_pka.py` script — the reusable `umapka`
   package itself had no way to predict in anything but water. Water
   still uses the dedicated aqueous regressor (best accuracy, MAE 0.994
   scaffold-split); any other solvent routes through
   `multisolvent_tuned.pkl` with the two solvent-descriptor features
   appended, in the *exact* encoding it was trained with.

3. **`umapka/solvents.py` — one canonical solvent table.** The eps/protic
   values used to build multisolvent features previously existed in
   three places (`solvent_features.py`, `predict_pka.py`,
   `tune_multisolvent.py`) with different numbers. `tune_multisolvent.py`
   is the actual training script, so its table is the one that's
   correct for `multisolvent_tuned.pkl` — everything else now imports
   from `umapka/solvents.py`, which copies that table verbatim.
   `solvent_features.py` at the repo root is left in place (marked
   deprecated) so nothing that imported it breaks, but nothing new uses
   it.

4. **Molarity / salt correction is now reachable from the CLI.**
   `PkaPredictor.predict(..., salt=, salt_concentration=)` already
   existed and worked, but `predict_pka.py` never exposed it — no
   `--salt`/`--molarity` flags at all. Added those, plus `--list-salts`.
   This correction (`umapka/solvation.py`) is a physics-based
   Debye-Hückel/Davies/Bjerrum calculation, not trained on any
   concentration-dependent pKa data — that's intentional (see the
   README section below on why).

5. **New: `umapka/mixtures.py` for binary solvent mixtures.** Genuinely
   new capability - neither branch had this. See the design rationale
   in that file's docstring and in the README section below.

## What's still NOT done (be honest with yourself about these)

- **The mixture predictions are not validated against any real mixture
  data**, because no mixture pKa training data exists in either repo.
  They're a physically-motivated interpolation between two *trained,
  validated* pure-solvent endpoints, not a third trained model. Treat
  results as directional, especially outside the water-rich region.
- **The salt/ionic-strength correction is still uncalibrated physics**,
  same as before this merge — nothing here changed that, since doing so
  would require real concentration-dependent pKa shift data to train
  against (see README).
- `NMP` and `EthyleneGlycol` have model predictions but no held-out MAE
  and no ionic-strength model (`solvation.py` doesn't cover them) -
  `--list-solvents` reports this per-solvent so it's not hidden.
- No retraining was done. If you get real experimental data for
  mixtures or concentration-dependent shifts, that's the actual fix -
  see the README section "If you get real data" for where it would
  plug in.
