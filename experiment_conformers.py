"""Test-time conformer averaging: does averaging predictions over K
independent 3D conformers beat committing to one?

THE ARGUMENT
Every prediction in this pipeline rests on ONE ETKDG conformer per
protonation state (seed=42). A single arbitrary conformer is a noisy
sample of a molecule's real conformational ensemble, and UMA's
embeddings depend on 3D geometry. If geometry noise is a meaningful part
of the residual, averaging K predictions cuts that component by ~sqrt(K).

WHY THIS IS DIFFERENT FROM WHAT ALREADY FAILED
Nine previous attempts changed what the model SEES (features, data,
pooling, architecture) or what it MINIMISES (L1/Huber - measured worse).
This changes neither. It is pure variance reduction on the inference
side, needs no retraining, and attacks NOISE rather than BIAS.

It also avoids the train/test mismatch that would come from averaging
FEATURES: each conformer produces a complete, valid prediction from the
model exactly as trained, and only the predictions are averaged.

Test-time only - no training features are touched, so this is safe to
evaluate directly on the held-out sets.

    python experiment_conformers.py --k 5
"""
import argparse
import time

import joblib
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="conformers per state")
    ap.add_argument("--model", default="models/model_core_v20_ensemble.pkl")
    a = ap.parse_args()

    from umapka import PkaPredictor, electronic
    from umapka.predictor import (protonation_pair_site_tagged,
                                   _smiles_to_atoms_with_site)

    p = PkaPredictor("models/model_core_v3.pkl")
    bundle = joblib.load(a.model)
    ext = joblib.load("feat_external_learned.pkl")

    def feats(prot, pi_, dep, di_, seed):
        """build_hybrid_features with an explicit conformer seed.
        elec_desc is topology-based, so it is seed-independent - only the
        UMA blocks change between conformers."""
        ap_, sp, mp = _smiles_to_atoms_with_site(prot, pi_, seed=seed)
        ad_, sd, md = _smiles_to_atoms_with_site(dep, di_, seed=seed)
        ep, ed = p.embeddings(ap_), p.embeddings(ad_)
        hg_p, hl_p = p.pool(ep), p.pool_local_multiscale(ep, sp, mp)
        hg_d, hl_d = p.pool(ed), p.pool_local_multiscale(ed, sd, md)
        g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
        l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
        dp = electronic.elec_desc(prot, pi_)
        dd = electronic.elec_desc(dep, di_)
        if dp is None or dd is None:
            raise RuntimeError("elec_desc failed")
        return np.nan_to_num(
            np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)

    seeds = [42 + 1000 * i for i in range(a.k)]   # 42 first = reproduces production
    for ds, rows in ext.items():
        print(f"\n=== {ds} (n={len(rows)}, K={a.k} conformers) ===", flush=True)
        y, per_seed, ok = [], [[] for _ in seeds], []
        t0 = time.time()
        for j, r in enumerate(rows):
            smi = r["smiles"]
            try:
                pr, pi_, dp_, di_, kind = protonation_pair_site_tagged(
                    smi, return_kind=True)
                vals = []
                for s in seeds:
                    vals.append(electronic.score_any(
                        bundle, feats(pr, pi_, dp_, di_, s)))
            except Exception:
                continue
            y.append(r["exp"])
            for i, v in enumerate(vals):
                per_seed[i].append(v)
            ok.append(smi)
            if (j + 1) % 50 == 0:
                el = (time.time() - t0) / 60
                print(f"  [{j+1}/{len(rows)}] {el:.1f} min "
                      f"(~{el/(j+1)*(len(rows)-j-1):.0f} min left)", flush=True)

        y = np.array(y)
        P = np.array(per_seed)                      # (K, n)
        e1 = np.abs(P[0] - y)                        # single conformer (production)
        eK = np.abs(P.mean(0) - y)                   # averaged over K
        print(f"  single conformer (seed 42) MAE {e1.mean():.4f}")
        print(f"  mean over {a.k} conformers    MAE {eK.mean():.4f}"
              f"   ({eK.mean()-e1.mean():+.4f})")
        # median is more robust to a single bad embedding
        eM = np.abs(np.median(P, axis=0) - y)
        print(f"  median over {a.k} conformers  MAE {eM.mean():.4f}"
              f"   ({eM.mean()-e1.mean():+.4f})")
        d = e1 - eK
        rng = np.random.default_rng(0)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                       for _ in range(3000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  paired delta {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  -> {'REAL' if lo > 0 else 'noise (CI spans 0)'}")
        spread = P.std(0)
        print(f"  per-molecule conformer spread: median {np.median(spread):.3f}, "
              f"p90 {np.percentile(spread,90):.3f} pKa units")
        print(f"  (spread is how much the prediction moves with geometry alone -"
              f" an upper bound on what averaging can recover)")
        joblib.dump({"y": y, "P": P, "smiles": ok}, f"conformer_{ds}.pkl")

    print("\n  v20 single-conformer baseline: novartis 0.918 | avlilumove 0.411")


if __name__ == "__main__":
    main()
