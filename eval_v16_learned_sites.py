"""Real-world (non-oracle) external eval for v16 (UMA + electronic
descriptors, hybrid GBM+ridge head): uses the LEARNED site finder
(umapka.site_finder, wired into protonation_pair_site_tagged) to pick
the site, not the ChemAxon marvin_atom ground truth eval_v16.py uses.

eval_v16.py reports the CEILING if site-finding were perfect (0.845
Novartis, per release/README.md) - real inference can't see marvin_atom.
This measures what users actually get.
"""
import sys, numpy as np, pandas as pd, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, ".")
from umapka import PkaPredictor
from umapka.predictor import protonation_pair_site_tagged
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.EState import EStateIndices


def elec_desc(smi, site_idx):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or site_idx is None or site_idx >= mol.GetNumAtoms(): return None
    try: AllChem.ComputeGasteigerCharges(mol)
    except Exception: return None
    q = np.nan_to_num(np.array([float(a.GetPropsAsDict().get("_GasteigerCharge", 0.0))
                                 for a in mol.GetAtoms()]), nan=0.0, posinf=0.0, neginf=0.0)
    try: est = np.array(EStateIndices(mol))
    except Exception: est = np.zeros(mol.GetNumAtoms())
    dm = Chem.GetDistanceMatrix(mol)
    s1 = np.where(dm[site_idx] <= 1)[0]; s2 = np.where(dm[site_idx] <= 2)[0]
    s3 = np.where(dm[site_idx] <= 3)[0]; a = mol.GetAtomWithIdx(site_idx)
    return np.array([q[site_idx], est[site_idx],
        q[s1].mean(), q[s1].min(), q[s1].max(), est[s1].mean(),
        q[s2].mean(), q[s2].min(), q[s2].max(), est[s2].mean(),
        q[s3].mean(), q[s3].min(), q[s3].max(), est[s3].mean(),
        q.mean(), q.min(), q.max(), q.std(),
        float(a.GetDegree()), float(a.GetTotalNumHs()), float(a.GetFormalCharge()),
        float(a.GetIsAromatic()), float(a.IsInRing()), float(a.GetAtomicNum()),
        Descriptors.TPSA(mol), Crippen.MolLogP(mol), float(Chem.GetFormalCharge(mol))], dtype=float)


b = joblib.load("models/model_core_v16_elec.pkl")
gbm, ridge, scaler, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]
print("loading UMA...")
p = PkaPredictor("models/model_core_v3.pkl")

rows = []
for path, ds in [
    ("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis"),
    ("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf", "avlilumove"),
]:
    mols = [m for m in Chem.ForwardSDMolSupplier(path) if m is not None]
    print(f"\n{ds}: {len(mols)}")
    for mol in tqdm(mols):
        if not (mol.HasProp("pKa") and mol.HasProp("marvin_atom")): continue
        try:
            exp = float(mol.GetProp("pKa")); ma = int(float(mol.GetProp("marvin_atom")))
            if not (0 < exp < 14): continue
            smi = Chem.MolToSmiles(mol)
            prot, pi_, dep, di_, kind = protonation_pair_site_tagged(smi, return_kind=True)
            hg_p, hl_p = p.state_features_v4(prot, pi_, kind, n_confs_base=1)
            hg_d, hl_d = p.state_features_v4(dep, di_, kind, n_confs_base=1)
            g_ = np.concatenate([hg_p, hg_d, hg_p - hg_d])
            l_ = np.concatenate([hl_p, hl_d, hl_p - hl_d])
            dp = elec_desc(prot, pi_); dd = elec_desc(dep, di_)
            if dp is None or dd is None: continue
            feat = np.nan_to_num(np.concatenate([g_, l_, dp, dd, dp - dd])).reshape(1, -1)
            raw = (1-bw)*gbm.predict(feat)[0] + bw*ridge.predict(scaler.transform(feat))[0]
            pred = float(cal.predict([raw])[0])
        except Exception:
            continue
        rows.append({"dataset": ds, "exp": exp, "pred": pred, "err": abs(pred-exp)})

d = pd.DataFrame(rows)
d.to_csv("characterization_external_v16_learned_sites.csv", index=False)
print(f"\n=== v16 + LEARNED SITE FINDER (n={len(d)}) ===")
print(f"MAE = {d.err.mean():.3f}")
print("\n=== BY DATASET (real-world number - compare to eval_v16.py's oracle 0.845/0.42) ===")
print(d.groupby("dataset")["err"].agg(["mean", "count"]).round(3))
