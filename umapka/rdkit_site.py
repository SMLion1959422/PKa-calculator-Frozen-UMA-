"""
umapka.rdkit_site - GPU-free site-level pKa model (RDKit features + LightGBM).

A cheap intrinsic micro-pKa predictor with the same role as
``PkaPredictor.predict_site``: one number per titratable site, all
other sites neutral. It exists for two reasons:

  * a baseline that any UMA model has to beat. In dev/site_ablation.py
    it reaches Novartis MAE 1.08 fully automatic, vs 1.17 for the
    shipped UMA model_core_v2;
  * an intrinsic-pKa source for ``umapka.microstates`` and the
    benchmarks on machines without a GPU or access to the gated UMA
    weights.

Features: a whole-molecule Morgan count vector (radius 2, 1024 bits)
plus site-centred environment counts (radius 0-3 rooted at the site
atom), element counts in topological shells 1-5 around the site, and a
few site-atom descriptors. Training uses the ANNOTATED site
(``marvin_atom``) of the Baltruschat & Czodrowski SDFs, not the SMARTS
priority site.
"""

from __future__ import annotations
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg

from .sites import Site, find_sites, neutralize

__all__ = ["global_features", "site_features", "pair_features",
           "RDKitSitePredictor"]

_NB = 512
_GLOBAL = rfg.GetMorganGenerator(radius=2, fpSize=1024)
_ROOTED = [rfg.GetMorganGenerator(radius=r, fpSize=_NB) for r in range(4)]
_ELEM = {"C": 0, "N": 1, "O": 2, "S": 3, "F": 4}


def global_features(mol: Chem.Mol) -> np.ndarray:
    return _GLOBAL.GetCountFingerprintAsNumPy(mol).astype(float)


def site_features(mol: Chem.Mol, atom: int, kind: str) -> np.ndarray:
    """Site-centred features; ``kind`` is "acid" or "base"."""
    out = [g.GetCountFingerprintAsNumPy(mol, fromAtoms=[atom]).astype(float)
           for g in _ROOTED]
    dm = Chem.GetDistanceMatrix(mol)[atom]
    shells = np.zeros((5, 6))
    for a in mol.GetAtoms():
        d = int(dm[a.GetIdx()])
        if 1 <= d <= 5:
            shells[d - 1, _ELEM.get(a.GetSymbol(), 5)] += 1
    at = mol.GetAtomWithIdx(atom)
    local = [at.GetIsAromatic(), at.GetTotalNumHs(), at.GetDegree(),
             at.GetAtomicNum(), kind == "acid"]
    return np.concatenate(out + [shells.ravel(), local])


def pair_features(mol: Chem.Mol, atom: int, kind: str) -> np.ndarray:
    return np.concatenate([global_features(mol), site_features(mol, atom, kind)])


class RDKitSitePredictor:
    """Site-level pKa regressor. ``fit_sdf`` trains it; ``predict_site``
    and ``intrinsic_pkas`` score sites of new molecules."""

    def __init__(self, **lgb_params):
        self.params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           verbose=-1, random_state=0)
        self.params.update(lgb_params)
        self.model = None
        self.n_train = 0

    # -- training ------------------------------------------------------
    @staticmethod
    def load_sdf(path: str, exclude_inchikeys: set | None = None):
        """(features, labels, n_excluded) from an SDF with pKa,
        marvin_atom and marvin_pKa_type properties."""
        RDLogger.DisableLog("rdApp.*")
        X, y, n_excl = [], [], 0
        for m in Chem.SDMolSupplier(path):
            if m is None:
                continue
            p = m.GetPropsAsDict()
            n = neutralize(m)
            if exclude_inchikeys and Chem.MolToInchiKey(n)[:14] in exclude_inchikeys:
                n_excl += 1
                continue
            kind = "acid" if p["marvin_pKa_type"] == "acidic" else "base"
            X.append(pair_features(n, int(p["marvin_atom"]), kind))
            y.append(float(p["pKa"]))
        return np.array(X), np.array(y), n_excl

    def fit(self, X, y):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(**self.params).fit(X, y)
        self.n_train = len(y)
        return self

    def fit_sdf(self, path: str, exclude_inchikeys: set | None = None):
        X, y, self.n_excluded = self.load_sdf(path, exclude_inchikeys)
        return self.fit(X, y)

    # -- prediction ----------------------------------------------------
    def predict_site(self, mol: Chem.Mol, site: Site) -> float:
        f = pair_features(mol, site.atom, site.kind).reshape(1, -1)
        return float(self.model.predict(np.ascontiguousarray(f))[0])

    def intrinsic_pkas(self, smiles: str) -> dict[int, float]:
        """site.index -> intrinsic pKa for every detected site, in the
        form ``umapka.microstates.build_model`` accepts."""
        mol = neutralize(Chem.MolFromSmiles(smiles))
        return {s.index: self.predict_site(mol, s) for s in find_sites(mol)}

    def save(self, path: str):
        import joblib
        joblib.dump({"model": self.model, "params": self.params,
                     "n_train": self.n_train, "feature": "rdkit_site_v1"}, path)

    @classmethod
    def load(cls, path: str) -> "RDKitSitePredictor":
        import joblib
        b = joblib.load(path)
        p = cls(**b["params"])
        p.model, p.n_train = b["model"], b["n_train"]
        return p
