"""TIER 2 step 1/2: cache PER-ATOM UMA embeddings (not pooled) for the
training set, so a learned attention-pooling head can be trained on top
(see train_attention_head.py).

WHY THIS AND NOT END-TO-END UMA FINE-TUNING: fine-tuning UMA itself is
the textbook fix (that is what Uni-pKa/Starling effectively do), but UMA
is a large equivariant GNN and this machine is CPU-only (torch 2.8.0+cpu,
no CUDA) - that is a GPU-scale job, not a CPU one.

What IS reachable on CPU is removing the *fixed* pooling bottleneck.
Today every model here collapses per-atom embeddings with mean/max over
the whole molecule (pool()) or a hand-chosen 2-bond shell
(pool_local()). RESULTS.md documents the consequence directly: "global
mean-pooling dilutes local pKa signal on large molecules", error growing
monotonically with molecule size. A learned attention pooling lets the
model decide per molecule which atoms carry the pKa signal, instead of
us hardcoding a radius. That is the representation-learning step frozen
pooling denies, and the head is small enough to train on CPU.

Stores float16 to keep the cache ~100-200 MB rather than ~450 MB; the
embeddings feed a small net, so fp16 storage precision is not the
limiting factor here.

Output: feat_atomwise.pkl
  {smiles: {"prot": {"emb": (n,128) f16, "dist": (n,) i8, "site": int},
            "dep":  {...},
            "kind": "acid"|"base", "pKa": float}}

Resumable: re-running skips molecules already in the cache, so an
interrupted run continues instead of starting over.
"""
import os
import time

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools

from umapka import PkaPredictor
from umapka.predictor import (protonation_pair_site_tagged,
                               _smiles_to_atoms_with_site)

RDLogger.DisableLog("rdApp.*")

OUT = "feat_atomwise.pkl"
SDF = "mlpka/datasets/combined_training_datasets_unique.sdf"


def state_atomwise(p, smi, site_idx):
    """Per-atom embeddings + topological distance to the titratable atom."""
    atoms, s_idx, mol_h = _smiles_to_atoms_with_site(smi, site_idx)
    emb = p.embeddings(atoms)
    dm = Chem.GetDistanceMatrix(mol_h)
    dist = np.clip(dm[s_idx], 0, 20).astype(np.int8)
    n = min(len(emb), len(dist))
    return {"emb": np.asarray(emb[:n], dtype=np.float16),
            "dist": dist[:n],
            "site": int(s_idx)}


def main():
    cache = {}
    if os.path.exists(OUT):
        try:
            cache = joblib.load(OUT)
            print(f"resuming: {len(cache)} molecules already cached")
        except Exception:
            cache = {}

    print("loading labels...")
    df = PandasTools.LoadSDF(SDF)
    pk_col = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
    todo = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None:
            continue
        try:
            v = float(r[pk_col])
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if not (0 < v < 14) or smi in cache:
            continue
        todo.append((smi, v))
    # de-dup while preserving order
    seen = set()
    todo = [(s, v) for s, v in todo if not (s in seen or seen.add(s))]
    print(f"to embed: {len(todo)}")

    print("loading UMA...")
    p = PkaPredictor("models/model_core_v3.pkl")

    t0 = time.time()
    n_done = n_fail = 0
    for i, (smi, pka) in enumerate(todo):
        try:
            prot, pi_, dep, di_, kind = protonation_pair_site_tagged(
                smi, return_kind=True)
            cache[smi] = {"prot": state_atomwise(p, prot, pi_),
                          "dep": state_atomwise(p, dep, di_),
                          "kind": kind, "pKa": pka}
            n_done += 1
        except Exception:
            n_fail += 1
        if (i + 1) % 250 == 0:
            el = (time.time() - t0) / 60
            rate = (i + 1) / max(el, 1e-9)
            print(f"  [{i+1}/{len(todo)}]  {el:.1f} min  "
                  f"~{(len(todo)-i-1)/max(rate,1e-9):.0f} min left  "
                  f"(ok={n_done} fail={n_fail})", flush=True)
            joblib.dump(cache, OUT, compress=0)

    joblib.dump(cache, OUT, compress=0)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"\ncached {len(cache)} molecules (ok={n_done} fail={n_fail}) "
          f"-> {OUT} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
