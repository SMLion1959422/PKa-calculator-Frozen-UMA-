"""Classical electronic descriptors (Gasteiger charge, EState index,
shell-averaged neighborhood stats) computed around one atom. Paired
with frozen UMA embeddings in the v16/v17 "hybrid" models
(models/model_core_v16_elec.pkl etc.) - ablation_xtb.py found UMA +
these descriptors beats UMA alone (0.537 -> 0.485 OOF) and matches real
GFN2-xTB quantum descriptors closely enough not to need an xtb install.

Shared here rather than duplicated across predict_ladder.py,
predict_microstates.py, and the eval_v16*/ablation scripts that all
need the exact same feature layout to match what the model was trained on.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices


def elec_desc(smiles: str, site_idx: int | None) -> np.ndarray | None:
    """27-dim electronic descriptor vector centered on `site_idx`, or
    None if the SMILES/index is invalid. Order MUST match the training
    scripts (train_v16_elec.py etc.) exactly - this is a fixed schema,
    not something to reorder for readability."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or site_idx is None or site_idx >= mol.GetNumAtoms():
        return None
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None
    q = np.nan_to_num(np.array([
        (float(a.GetDoubleProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0)
        for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        est = np.array(EStateIndices(mol))
    except Exception:
        est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[site_idx] <= 1)[0]
    s2 = np.where(dm[site_idx] <= 2)[0]
    s3 = np.where(dm[site_idx] <= 3)[0]
    a = mol.GetAtomWithIdx(site_idx)
    return np.array([q[site_idx], est[site_idx],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()), float(a.GetFormalCharge()),
        float(a.GetIsAromatic()), float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol), float(Chem.GetFormalCharge(mol))],
        dtype=float)


def build_hybrid_features(predictor, prot_smi: str, prot_idx: int,
                           dep_smi: str, dep_idx: int, kind: str,
                           n_confs_base: int = 1) -> np.ndarray | None:
    """The v16/v17 feature vector: UMA global + multiscale-local pooling
    for both protonation states and their difference, concatenated with
    elec_desc on both states and its difference. Returns shape (1, N),
    or None if either state's descriptors fail.

    Split out from predict_hybrid() so an evaluation harness can CACHE
    these vectors once (the UMA passes are the whole cost) and then
    score many candidate heads against them instantly, instead of
    re-embedding the test set per model variant.

    `n_confs_base` >1 averages the pooled embedding over that many
    conformers for BASE sites (see state_features_v4) - costs
    proportionally more UMA calls."""
    hg_p, hl_p = predictor.state_features_v4(prot_smi, prot_idx, kind,
                                             n_confs_base=n_confs_base)
    hg_d, hl_d = predictor.state_features_v4(dep_smi, dep_idx, kind,
                                             n_confs_base=n_confs_base)
    g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
    l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
    dp = elec_desc(prot_smi, prot_idx)
    dd = elec_desc(dep_smi, dep_idx)
    if dp is None or dd is None:
        return None
    return np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)


def score_hybrid(bundle: dict, feat: np.ndarray) -> float:
    """Score ONE prebuilt feature vector, shape (1, N), with a hybrid
    bundle ({"gbm","ridge","scaler","blend_w","calibrator"}): blended
    GBM+Ridge, isotonic-calibrated.

    Single-row only, and it now says so loudly. This used to index [0]
    on the prediction array, so passing a whole (n, N) matrix silently
    scored only the first row and broadcast that one number against
    every label - which produced a plausible-looking but meaningless
    MAE (v16 "scoring" 2.235 on Novartis instead of 1.001). Use
    score_hybrid_batch for matrices.
    """
    feat = np.asarray(feat)
    if feat.ndim != 2 or feat.shape[0] != 1:
        raise ValueError(
            f"score_hybrid expects one row, shape (1, N); got {feat.shape}. "
            f"Use score_hybrid_batch() for multiple rows."
        )
    return float(score_hybrid_batch(bundle, feat)[0])


def score_hybrid_batch(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Vectorized hybrid scoring over an (n, N) matrix."""
    gbm, ridge, sc, bw, cal = (bundle["gbm"], bundle["ridge"], bundle["scaler"],
                               bundle["blend_w"], bundle["calibrator"])
    X = np.ascontiguousarray(np.nan_to_num(np.asarray(X, dtype=np.float64),
                                            nan=0.0, posinf=0.0, neginf=0.0))
    raw = (1 - bw) * gbm.predict(X) + bw * ridge.predict(sc.transform(X))
    return np.asarray(cal.predict(raw), dtype=float)


def predict_hybrid(predictor, bundle: dict, prot_smi: str, prot_idx: int,
                    dep_smi: str, dep_idx: int, kind: str) -> float | None:
    """Score one protonated/deprotonated pair with a v16/v17-style
    hybrid bundle. Returns None if either state's descriptors fail."""
    feat = build_hybrid_features(predictor, prot_smi, prot_idx,
                                 dep_smi, dep_idx, kind)
    if feat is None:
        return None
    return score_hybrid(bundle, feat)


def is_hybrid_bundle(obj) -> bool:
    """v16/v17 format: a single GBM blended with a ridge model."""
    return isinstance(obj, dict) and {"gbm", "ridge", "scaler", "blend_w", "calibrator"} <= obj.keys()


def is_ensemble_bundle(obj) -> bool:
    """v20/v21 format: a list of bagged tree models (+ optional HistGB)
    blended with ridge."""
    return (isinstance(obj, dict) and obj.get("kind") == "v20_ensemble"
            and {"lgb", "ridge", "scaler", "w_ridge", "calibrator"} <= obj.keys())


def score_ensemble_batch(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Vectorized scoring for a v20/v21 ensemble bundle: average the
    bagged tree members, optionally blend in ridge, then calibrate.

    Lives here rather than in the training script so inference never has
    to import a module that trains models on import.
    """
    X = np.ascontiguousarray(np.nan_to_num(np.asarray(X, dtype=np.float64),
                                            nan=0.0, posinf=0.0, neginf=0.0))
    parts = [m.predict(X) for m in bundle["lgb"]]
    if bundle.get("hgb"):
        parts += [m.predict(X) for m in bundle["hgb"]]
    trees = np.mean(parts, axis=0)
    w = bundle["w_ridge"]
    raw = trees if not w else (1 - w) * trees + w * bundle["ridge"].predict(
        bundle["scaler"].transform(X))
    return np.asarray(bundle["calibrator"].predict(raw), dtype=float)


def score_any(bundle: dict, feat: np.ndarray) -> float:
    """Score ONE feature row with whichever bundle format is given."""
    feat = np.asarray(feat).reshape(1, -1)
    if is_ensemble_bundle(bundle):
        return float(score_ensemble_batch(bundle, feat)[0])
    if is_hybrid_bundle(bundle):
        return float(score_hybrid_batch(bundle, feat)[0])
    raise ValueError(f"unrecognized model bundle: keys={sorted(bundle)[:8]}")


def score_any_batch(bundle: dict, X: np.ndarray) -> np.ndarray:
    if is_ensemble_bundle(bundle):
        return score_ensemble_batch(bundle, X)
    if is_hybrid_bundle(bundle):
        return score_hybrid_batch(bundle, X)
    raise ValueError(f"unrecognized model bundle: keys={sorted(bundle)[:8]}")
