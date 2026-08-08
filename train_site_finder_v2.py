"""LEARNED SITE-FINDER v2 - ranks candidate ATOMS, not site-type names.

WHY v1 FAILED: it chose between site TYPE labels (pyridine_N vs
tert_amine). But verify_marvin_atoms.py showed acid/base KIND already
agrees 95.3% on Novartis while ATOM agrees only 79% - so the dominant
error is picking the wrong atom among several of the SAME type. v1
could not fix that even in principle, and its overrides scored 1.773.

WHAT IS DIFFERENT:
 - Ground truth is marvin_atom (real ChemAxon annotation), NOT my
   hand-written SITE_RANGES. v1's targets were a function of the label,
   so it could score 84.9% by learning site-type frequencies without
   reading the molecule at all - then fired that prior blindly at test
   time.
 - Candidates are individual ATOMS. Every atom matching any ionizable
   SMARTS becomes a row.
 - Per-atom features: element, charge, H count, hybridisation,
   aromaticity, ring membership, Gasteiger charge, EState index, plus
   the same for its 1- and 2-bond neighbourhood, plus which SMARTS
   patterns matched that atom, plus molecule-level context.
 - Ranked with LambdaRank (groups = molecules), which optimises "is the
   right atom on top" rather than per-atom classification accuracy.

Trains on molecules WITH marvin_atom, so the target is independent of
the pKa value - no circularity."""
import sys
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices
from tqdm import tqdm
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka.predictor import ACID_SITES, BASE_SITES, neutralize

ALL_PATTS = ([(n, s, ai, "acid") for n, s, ai in ACID_SITES] +
             [(n, s, ai, "base") for n, s, ai in BASE_SITES])
PATT_NAMES = [p[0] for p in ALL_PATTS]
PATT_IDX = {n: i for i, n in enumerate(PATT_NAMES)}
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D]


def candidate_atoms(mol):
    """Every ATOM matched by any ionizable SMARTS, with which patterns hit it."""
    hits = {}
    for name, smarts, ai, kind in ALL_PATTS:
        pt = Chem.MolFromSmarts(smarts)
        if pt is None:
            continue
        for m in mol.GetSubstructMatches(pt):
            idx = m[ai]
            if idx not in hits:
                hits[idx] = {"patterns": set(), "kinds": set()}
            hits[idx]["patterns"].add(name)
            hits[idx]["kinds"].add(kind)
    return hits


def atom_features(mol, idx, info, gast, est, dm, mol_ctx, n_cands, prio_rank):
    a = mol.GetAtomWithIdx(idx)
    patt_vec = [0.0] * len(PATT_NAMES)
    for p in info["patterns"]:
        if p in PATT_IDX:
            patt_vec[PATT_IDX[p]] = 1.0
    s1 = np.where(dm[idx] <= 1)[0]
    s2 = np.where(dm[idx] <= 2)[0]
    hyb = [1.0 if a.GetHybridization() == h else 0.0 for h in HYB]
    return patt_vec + hyb + mol_ctx + [
        float(a.GetAtomicNum()), float(a.GetFormalCharge()),
        float(a.GetTotalNumHs()), float(a.GetDegree()),
        float(a.GetIsAromatic()), float(a.IsInRing()),
        float(a.GetTotalValence()),
        gast[idx], est[idx],
        gast[s1].mean(), gast[s1].min(), gast[s1].max(),
        gast[s2].mean(), gast[s2].min(), gast[s2].max(),
        est[s1].mean(), est[s2].mean(),
        float(len(s1) - 1), float(len(s2) - 1),
        1.0 if "acid" in info["kinds"] else 0.0,
        1.0 if "base" in info["kinds"] else 0.0,
        float(len(info["patterns"])), float(n_cands), float(prio_rank),
    ]


def build(path, need_truth=True):
    X, y, groups, meta = [], [], [], []
    gid = 0
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    for mol in tqdm(mols, desc=path.split("/")[-1][:28]):
        if need_truth and not mol.HasProp("marvin_atom"):
            continue
        try:
            ma = int(float(mol.GetProp("marvin_atom"))) if need_truth else -1
            smi = Chem.MolToSmiles(mol)
            nm = neutralize(Chem.Mol(mol))
        except Exception:
            continue
        if need_truth and (ma < 0 or ma >= nm.GetNumAtoms()):
            continue
        hits = candidate_atoms(nm)
        if not hits:
            continue
        if need_truth and ma not in hits:
            continue          # truth atom not even a candidate - unfixable here
        try:
            AllChem.ComputeGasteigerCharges(nm)
            gast = np.nan_to_num(np.array(
                [float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
                 for a in nm.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
            est = np.array(EStateIndices(nm))
            dm = Chem.GetDistanceMatrix(nm)
        except Exception:
            continue
        mol_ctx = [Descriptors.MolWt(nm), Crippen.MolLogP(nm),
                   Descriptors.TPSA(nm), float(Descriptors.RingCount(nm)),
                   float(Descriptors.NumAromaticRings(nm)),
                   float(nm.GetNumAtoms()), float(Chem.GetFormalCharge(nm))]
        ordered = sorted(hits.keys())
        for rank, idx in enumerate(ordered):
            X.append(atom_features(nm, idx, hits[idx], gast, est, dm,
                                    mol_ctx, len(ordered), rank))
            y.append(1 if (need_truth and idx == ma) else 0)
            groups.append(gid)
            meta.append({"smiles": smi, "atom": idx, "gid": gid,
                          "is_truth": (need_truth and idx == ma),
                          "n_cands": len(ordered)})
        gid += 1
    return (np.array(X, dtype=float), np.array(y),
            np.array(groups), pd.DataFrame(meta))


print("building TRAINING set (ground truth = marvin_atom)...")
Xtr, ytr, gtr, mtr = build("mlpka/datasets/combined_training_datasets_unique.sdf")
print(f"  {len(np.unique(gtr))} molecules, {len(Xtr)} candidate atoms, "
      f"{Xtr.shape[1]} features")
print(f"  mean candidates/molecule: {len(Xtr)/max(len(np.unique(gtr)),1):.2f}")

print("\n5-fold grouped ranking (LambdaRank)...")
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(ytr))
for i, (tr, va) in enumerate(gkf.split(Xtr, ytr, gtr)):
    gtr_tr = pd.Series(gtr[tr]).value_counts(sort=False).sort_index().values
    order = np.argsort(gtr[tr], kind="stable")
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400,
                        learning_rate=0.05, num_leaves=31, verbose=-1,
                        random_state=42)
    m.fit(Xtr[tr][order], ytr[tr][order], group=gtr_tr)
    oof[va] = m.predict(Xtr[va])
    print(f"  fold {i+1}/5")

mtr["score"] = oof
picked = mtr.loc[mtr.groupby("gid")["score"].idxmax()]
prio = mtr.groupby("gid", sort=False).head(1)
print("\n" + "=" * 62)
print("SITE-FINDING ACCURACY (out-of-fold, atom-level)")
print("=" * 62)
print(f"learned ranker  : {picked.is_truth.mean()*100:.1f}%")
print(f"SMARTS priority : {prio.is_truth.mean()*100:.1f}%")
multi = mtr[mtr.n_cands > 1]
pm = multi.loc[multi.groupby("gid")["score"].idxmax()]
pp = multi.groupby("gid", sort=False).head(1)
print(f"\nmulti-candidate molecules only (n={pm.gid.nunique()}):")
print(f"  learned ranker  : {pm.is_truth.mean()*100:.1f}%")
print(f"  SMARTS priority : {pp.is_truth.mean()*100:.1f}%")
print("\n  Novartis reference: SMARTS priority gets 79.0% atom accuracy.")
print("  If the learned ranker is well above that here, it should")
print("  close part of the 0.998 -> 0.845 site-assignment gap.")

full_group = pd.Series(gtr).value_counts(sort=False).sort_index().values
order = np.argsort(gtr, kind="stable")
final = lgb.LGBMRanker(objective="lambdarank", n_estimators=400,
                        learning_rate=0.05, num_leaves=31, verbose=-1,
                        random_state=42)
final.fit(Xtr[order], ytr[order], group=full_group)
joblib.dump({"model": final, "patt_names": PATT_NAMES},
            "models/site_finder_v2.pkl")
print("\nsaved -> models/site_finder_v2.pkl")

print("\n" + "=" * 62)
print("HELD-OUT CHECK ON THE TEST SETS (site accuracy only, no pKa)")
print("=" * 62)
for path, label in [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]:
    Xt, yt, gt, mt = build(path)
    if len(Xt) == 0:
        continue
    mt["score"] = final.predict(Xt)
    pk = mt.loc[mt.groupby("gid")["score"].idxmax()]
    pr = mt.groupby("gid", sort=False).head(1)
    mm = mt[mt.n_cands > 1]
    pkm = mm.loc[mm.groupby("gid")["score"].idxmax()]
    prm = mm.groupby("gid", sort=False).head(1)
    print(f"\n{label} (n={mt.gid.nunique()} molecules)")
    print(f"  learned ranker  : {pk.is_truth.mean()*100:.1f}%")
    print(f"  SMARTS priority : {pr.is_truth.mean()*100:.1f}%")
    print(f"  multi-candidate only (n={pkm.gid.nunique()}): "
          f"learned {pkm.is_truth.mean()*100:.1f}% vs "
          f"priority {prm.is_truth.mean()*100:.1f}%")
