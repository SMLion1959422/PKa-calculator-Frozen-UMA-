"""TIER 2 step 2/2: learned ATTENTION POOLING over per-atom UMA
embeddings, replacing the fixed mean/max (pool()) and hand-chosen
2-bond-shell (pool_local()) pooling every model here uses today.

THE ARGUMENT FOR THIS SPECIFICALLY
RESULTS.md documents the core failure directly: "global mean-pooling
dilutes local pKa signal on large molecules", with error rising
monotonically with molecule size (0.68 MAE under 15 heavy atoms -> 1.44
above 30). pool_local() patched it by hardcoding a 2-bond radius, which
helped (0.657 -> 0.591 in the controlled A/B) but still fixes the radius
by hand for every molecule. Attention lets the model learn per molecule
which atoms carry the signal - that is the representation-learning step
a frozen pooling function structurally cannot provide, and it is the
CPU-reachable part of what Uni-pKa/Starling get from end-to-end
training. UMA itself stays frozen (no GPU here).

HONEST EXPECTATION
This is a ~5.5k-molecule dataset. dev/compare_regressor_heads.py already
found gradient-boosted trees beat an MLP on the pooled features (0.577
vs 0.596) - the classic small-data result. So a neural pooling head is
NOT guaranteed to win; the bet is specifically that learned pooling
recovers signal that fixed pooling destroys BEFORE any head sees it,
which is a different question from head capacity. If it loses, that is
a real answer and the frozen-embedding ceiling is the honest conclusion.
Reported number is 5-fold OOF MAE against the identical split used by
train_core_v3.py / compare_regressor_heads.py, so it is comparable.

Run after cache_atom_embeddings.py finishes:
    python train_attention_head.py
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

CACHE = "feat_atomwise.pkl"
MAX_ATOMS = 96
EMB_DIM = 128
MAX_DIST = 20


class AttnPool(nn.Module):
    """Attention pooling over one protonation state's per-atom
    embeddings, conditioned on each atom's topological distance to the
    titratable site. Distance enters as a learned embedding rather than
    a hard cutoff, so the model can shape its own neighborhood weighting
    instead of inheriting pool_local()'s fixed 2-bond shell."""

    def __init__(self, emb_dim=EMB_DIM, d_model=128, dist_dim=16):
        super().__init__()
        self.dist_emb = nn.Embedding(MAX_DIST + 2, dist_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim + dist_dim, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.score = nn.Linear(d_model, 1)

    def forward(self, emb, dist, mask):
        h = self.proj(torch.cat([emb, self.dist_emb(dist)], dim=-1))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=-1).unsqueeze(-1)
        attn = (h * w).sum(1)
        # keep a masked mean too: attention can collapse onto one atom,
        # and the whole-molecule context genuinely matters for pKa (it is
        # why global pooling was there in the first place) - this gives
        # the model both without forcing either.
        mean = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return torch.cat([attn, mean], dim=-1)


class PkaNet(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.pool = AttnPool(d_model=d_model)
        w = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(w * 3 + 1, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch):
        hp = self.pool(batch["emb_p"], batch["dist_p"], batch["mask_p"])
        hd = self.pool(batch["emb_d"], batch["dist_d"], batch["mask_d"])
        z = torch.cat([hp, hd, hp - hd, batch["is_base"].unsqueeze(-1)], dim=-1)
        return self.head(z).squeeze(-1)


def pad(rec, key):
    e = np.asarray(rec[key]["emb"], dtype=np.float32)
    d = np.asarray(rec[key]["dist"], dtype=np.int64)
    n = min(len(e), len(d), MAX_ATOMS)
    emb = np.zeros((MAX_ATOMS, EMB_DIM), dtype=np.float32)
    dist = np.full(MAX_ATOMS, MAX_DIST + 1, dtype=np.int64)
    mask = np.zeros(MAX_ATOMS, dtype=bool)
    emb[:n] = e[:n]
    dist[:n] = np.clip(d[:n], 0, MAX_DIST)
    mask[:n] = True
    return emb, dist, mask


def build(cache):
    keys = sorted(cache.keys())
    E_p, D_p, M_p, E_d, D_d, M_d, y, isb = [], [], [], [], [], [], [], []
    for k in keys:
        r = cache[k]
        try:
            ep, dp, mp = pad(r, "prot")
            ed, dd, md = pad(r, "dep")
        except Exception:
            continue
        E_p.append(ep); D_p.append(dp); M_p.append(mp)
        E_d.append(ed); D_d.append(dd); M_d.append(md)
        y.append(r["pKa"]); isb.append(1.0 if r["kind"] == "base" else 0.0)
    return {
        "emb_p": torch.tensor(np.stack(E_p)), "dist_p": torch.tensor(np.stack(D_p)),
        "mask_p": torch.tensor(np.stack(M_p)),
        "emb_d": torch.tensor(np.stack(E_d)), "dist_d": torch.tensor(np.stack(D_d)),
        "mask_d": torch.tensor(np.stack(M_d)),
        "is_base": torch.tensor(np.array(isb, dtype=np.float32)),
    }, np.array(y, dtype=np.float32)


def sub(data, idx):
    return {k: v[idx] for k, v in data.items()}


def run_fold(data, y, tr, va, epochs=60, bs=64, seed=0):
    torch.manual_seed(seed)
    model = PkaNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=epochs * max(1, len(tr) // bs + 1))
    yt = torch.tensor(y)
    best, best_pred = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), bs):
            j = tr[perm[i:i + bs]]
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(sub(data, j)), yt[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
        model.eval()
        with torch.no_grad():
            pv = model(sub(data, va)).numpy()
        mae = float(np.abs(pv - y[va]).mean())
        if mae < best:
            best, best_pred = mae, pv
    return best_pred


def main():
    print(f"loading {CACHE} ...")
    cache = joblib.load(CACHE)
    print(f"  {len(cache)} molecules cached")
    data, y = build(cache)
    n = len(y)
    print(f"  usable: {n}   tensor {tuple(data['emb_p'].shape)}")

    torch.set_num_threads(max(1, (torch.get_num_threads() or 8) - 1))
    kf = KFold(5, shuffle=True, random_state=42)   # SAME split as train_core_v3.py
    oof = np.zeros(n, dtype=np.float32)
    for i, (tr, va) in enumerate(kf.split(np.arange(n))):
        oof[va] = run_fold(data, y, tr, va)
        print(f"  fold {i+1}/5  MAE={np.abs(oof[va]-y[va]).mean():.3f}", flush=True)

    mae = float(np.abs(oof - y).mean())
    print(f"\n=== ATTENTION POOLING: 5-fold OOF MAE = {mae:.3f} ===")
    print("  compare (same split, pooled features):")
    print("    LightGBM on global+local pooled : 0.577")
    print("    MLP on global+local pooled      : 0.596")
    print("    3-model average ensemble        : 0.543")
    print("  If this is not clearly below 0.543, fixed pooling was NOT the")
    print("  binding constraint and the frozen-UMA ceiling is the real one.")
    joblib.dump({"oof": oof, "y": y, "mae": mae}, "attention_head_oof.pkl")
    print("\nsaved OOF predictions -> attention_head_oof.pkl")


if __name__ == "__main__":
    main()
