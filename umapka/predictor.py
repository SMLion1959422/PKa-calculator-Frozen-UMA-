"""
umapka - pKa prediction from UMA foundation-model embeddings.

Uses Meta's UMA universal atomistic model as a frozen feature
extractor: per-atom embeddings are pulled from the input to the
energy head, pooled, and combined as paired protonated/deprotonated
difference features. A gradient-boosted regressor maps those to pKa.

Scope (please read before using):
  - Validated for MONOPROTIC acids and bases in the range pKa 2-12.
  - Also handles the FIRST ionization of simple polyprotic acids and
    bases in that range.
  - NOT reliable for: pKa2 and beyond, zwitterionic carboxyls
    (amino acids), or pKa outside 2-12.
See README for benchmark numbers and known limitations.
"""

from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

__all__ = ["PkaPredictor", "ACID_SITES", "BASE_SITES",
           "neutralize", "protonation_pair"]


# ---------------------------------------------------------------------
# Titratable-site definitions (SMARTS, atom index within the match)
# ---------------------------------------------------------------------
ACID_SITES = [
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 2),
    ("sulfonic_acid",   "[SX4](=O)(=O)[OX2H1]", 3),
    ("phosphoric_acid", "[PX4](=O)[OX2H1]", 2),
    ("tetrazole",       "c1nnn[nH]1", 0),
    ("tetrazole_2",     "c1nn[nH]n1", 0),
    ("sulfonamide_2",   "[SX4](=O)(=O)[NX3H1]", 3),
    ("sulfonamide_1",   "[SX4](=O)(=O)[NX3H2]", 3),
    ("thiol",           "[SX2H1]", 0),
    ("hydroxamic_acid",  "[CX3](=O)[NX3][OX2H1]", 3),
    ("phenol",          "[c][OX2H1]", 1),
    ("imide",           "[CX3](=O)[NX3H1][CX3]=O", 2),
]

BASE_SITES = [
    ("guanidine",  "[NX3][CX3](=[NX2])[NX3]", 2),
    ("amidine",    "[NX3][CX3]=[NX2]", 2),
    ("prim_amine", "[NX3;H2;!$(N[C,S]=[O,S,N]);!$(N-a)]", 0),
    ("sec_amine",  "[NX3;H1;!$(N[C,S]=[O,S,N]);!$(N-a)]", 0),
    ("tert_amine", "[NX3;H0;!$(N[C,S]=[O,S,N]);!$(N-a)]", 0),
    # pyridine-like only: aromatic N, no H, NOT adjacent to another
    # aromatic N (excludes tetrazole/triazole/imidazole ring nitrogens,
    # which are not basic in this sense)
    ("pyridine_N", "[nX2;H0;!$(n~n)]", 0),
    ("aniline",    "[NX3;H2]-a", 0),
    ("aniline_sec", "[NX3;H1]-a", 0),
    ("aniline_tert","[NX3;H0]-a", 0),
]

_NEUTRALIZE_PATTERN = Chem.MolFromSmarts(
    "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
)


# ---------------------------------------------------------------------
# molecule utilities
# ---------------------------------------------------------------------
def neutralize(mol: Chem.Mol) -> Chem.Mol:
    """Strip formal charges where chemically reasonable.

    Public pKa datasets frequently store molecules already ionized,
    which prevents the neutral-form SMARTS above from matching.
    """
    rw = Chem.RWMol(mol)
    for (idx,) in rw.GetSubstructMatches(_NEUTRALIZE_PATTERN):
        atom = rw.GetAtomWithIdx(idx)
        charge, n_h = atom.GetFormalCharge(), atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(n_h - charge)
        atom.SetNoImplicit(True)
        atom.UpdatePropertyCache(strict=False)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return mol


def _shift_hydrogen(mol, idx, d_h, d_charge):
    """Change hydrogen count and formal charge at one atom.

    Round-trips through SMILES so hydrogen bookkeeping stays correct
    for any subsequent SMARTS matching.
    """
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)
    n_h = atom.GetTotalNumHs() + d_h
    if n_h < 0:
        return None
    atom.SetNumExplicitHs(n_h)
    atom.SetNoImplicit(True)
    atom.SetFormalCharge(atom.GetFormalCharge() + d_charge)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        round_trip = Chem.MolFromSmiles(Chem.MolToSmiles(out))
        return round_trip if round_trip is not None else out
    except Exception:
        return None


def protonation_pair(smiles: str) -> tuple[str, str]:
    """Return (protonated_smiles, deprotonated_smiles) differing by one proton.

    Acidic sites are tried first (neutral acid -> anion), then basic
    sites (cation -> neutral base). The FIRST matching site by the
    priority order above is used; for molecules with several
    titratable groups this may not be the site of interest - use
    ``PkaPredictor.sites`` and ``predict_site`` to choose explicitly.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    mol = neutralize(mol)

    for _, smarts, ai in ACID_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            dep = _shift_hydrogen(mol, matches[0][ai], -1, -1)
            if dep is not None:
                return Chem.MolToSmiles(mol), Chem.MolToSmiles(dep)

    for _, smarts, ai in BASE_SITES:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            pro = _shift_hydrogen(mol, matches[0][ai], +1, +1)
            if pro is not None:
                return Chem.MolToSmiles(pro), Chem.MolToSmiles(mol)

    raise RuntimeError(f"no titratable site found in {smiles}")


def _smiles_to_atoms(smiles: str, seed: int = 42) -> Atoms:
    """SMILES -> single MMFF-optimized 3D conformer as an ASE Atoms.

    Net formal charge is read from the structure, never assigned by hand.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    charge = Chem.GetFormalCharge(mol)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"3D embedding failed for {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    atoms = Atoms(
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
    )
    atoms.info = {"charge": int(charge), "spin": 1}
    return atoms


# ---------------------------------------------------------------------
# predictor
# ---------------------------------------------------------------------
class PkaPredictor:
    """pKa predictor built on frozen UMA embeddings.

    Parameters
    ----------
    model_path : str
        Path to the trained regressor (joblib .pkl).
    uma_model : str
        UMA checkpoint name, e.g. ``"uma-s-1p1"``. Requires a
        HuggingFace account with access to ``facebook/UMA``.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Examples
    --------
    >>> p = PkaPredictor("model_core.pkl")
    >>> p.predict("CC(=O)O")          # acetic acid
    4.23
    """

    def __init__(self, model_path: str,
                 uma_model: str = "uma-s-1p1",
                 device: str | None = None,
                 free_energy_model_path: str | None = None,
                 multisolvent_model_path: str | None = "models/multisolvent_tuned.pkl"):
        import torch, joblib
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch

        predictor = pretrained_mlip.get_predict_unit(uma_model, device=device)
        self._calc = FAIRChemCalculator(predictor, task_name="omol")

        # the model is built lazily; one forward pass materializes it
        probe = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        probe.info = {"charge": 0, "spin": 1}
        probe.calc = self._calc
        probe.get_potential_energy()

        module = predictor.tracked_modules()["model"]
        self._energy_head = (
            module.module.output_heads["energyandforcehead"].head.energy_block
        )
        try:
            self._calc.use_cache = False   # caching would skip the forward pass
        except Exception:
            pass

        self._buffer = {}
        self.regressor = joblib.load(model_path)

        # OPTIONAL second regressor for predict_chain() - a DIFFERENT
        # model trained by train_free_energy_model.py on single-state
        # embedding differences. Not the same file as model_path above;
        # predict()/predict_site()/predict_all_sites() never touch this.
        self._free_energy_model = (
            joblib.load(free_energy_model_path)
            if free_energy_model_path else None
        )

        # OPTIONAL third regressor for non-aqueous solvents (`solvent=`
        # argument on predict()/predict_site()/predict_detailed()).
        # Loaded LAZILY on first non-water use, not here, so importing
        # this class doesn't require multisolvent_tuned.pkl to exist if
        # you only ever predict in water. See _load_multisolvent().
        self._multisolvent_model_path = multisolvent_model_path
        self._multisolvent_bundle = None

    # -- embedding extraction ------------------------------------------
    def _hook(self, module, inputs):
        if inputs and self._torch.is_tensor(inputs[0]):
            self._buffer["h"] = inputs[0].detach().float().cpu().numpy()

    def embeddings(self, atoms: Atoms) -> np.ndarray:
        """(n_atoms, 128) per-atom embeddings from the energy-head input."""
        self._buffer.clear()
        handle = self._energy_head[0].register_forward_pre_hook(self._hook)
        try:
            a = atoms.copy()
            a.calc = self._calc
            self._calc.reset()
            a.get_potential_energy()
        finally:
            handle.remove()
        emb = self._buffer.get("h")
        if emb is None or emb.shape[0] != len(atoms):
            raise RuntimeError("embedding extraction failed")
        return emb

    @staticmethod
    def pool(emb: np.ndarray) -> np.ndarray:
        """L2-normalize per atom, then concatenate mean and max pooling.

        Per-atom normalization matters: raw mean-pooling is dominated
        by a few high-magnitude atoms.
        """
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        return np.concatenate([norm.mean(0), norm.max(0)])

    def features(self, protonated: str, deprotonated: str) -> np.ndarray:
        """[h_prot ; h_deprot ; h_prot - h_deprot], shape (1, 768).

        pKa describes a transition rather than a molecule, so both
        charge states and their difference are encoded.
        """
        h_p = self.pool(self.embeddings(_smiles_to_atoms(protonated)))
        h_d = self.pool(self.embeddings(_smiles_to_atoms(deprotonated)))
        return np.concatenate([h_p, h_d, h_p - h_d]).reshape(1, -1)

    # -- multi-solvent support -------------------------------------------
    def _load_multisolvent(self):
        """Lazily load the multisolvent regressor bundle on first use.
        Not loaded in __init__ so predicting only-ever-in-water never
        requires multisolvent_tuned.pkl to exist.
        """
        if self._multisolvent_bundle is None:
            import joblib
            if not self._multisolvent_model_path:
                raise RuntimeError(
                    "no multisolvent_model_path configured - construct "
                    "PkaPredictor(..., multisolvent_model_path="
                    "'models/multisolvent_tuned.pkl') to enable solvent!='water'"
                )
            self._multisolvent_bundle = joblib.load(self._multisolvent_model_path)
        return self._multisolvent_bundle

    def _base_pka(self, pair_feat: np.ndarray, solvent: str) -> float:
        """Route a 768-dim pair feature through the right regressor for
        `solvent`. Water uses the dedicated aqueous regressor
        (self.regressor - the more accurate, water-only model); any
        other solvent uses the multisolvent regressor with the two
        extra solvent-descriptor features it was trained on, in the
        EXACT encoding tune_multisolvent.py used (see umapka.solvents -
        NOT raw physical constants).
        """
        from . import solvents as _solvents
        info = _solvents.resolve_solvent(solvent)
        if info.name == "Water":
            # model_core.pkl is a plain regressor. model_core_v2.pkl (the
            # RECOMMENDED aqueous model - RESULTS.md: leakage-fixed,
            # Novartis MAE 1.16 vs 1.41 for model_core.pkl) is instead a
            # {"regressor":..., "calibrator":...} bundle: raw regressor
            # output is passed through a calibrator before it's a real
            # pKa. Handle both shapes rather than silently mis-scoring
            # one of them.
            if isinstance(self.regressor, dict) and "regressor" in self.regressor:
                raw = float(self.regressor["regressor"].predict(pair_feat)[0])
                if "calibrator" in self.regressor and self.regressor["calibrator"] is not None:
                    return float(self.regressor["calibrator"].predict([raw])[0])
                return raw
            return float(self.regressor.predict(pair_feat)[0])
        bundle = self._load_multisolvent()
        feat = np.concatenate([pair_feat.reshape(-1), [info.eps_norm, info.protic]]).reshape(1, -1)
        return float(bundle["model"].predict(feat)[0])

    # -- prediction -----------------------------------------------------
    def predict(self, smiles: str, solvent: str = "water",
                salt: str | None = None,
                salt_concentration: float | None = None) -> float:
        """Predict pKa for the first titratable site found, in `solvent`
        (default water - the best-validated case, MAE 0.994 scaffold
        split). Other solvents route through the multisolvent regressor;
        see umapka.solvents.SOLVENTS for which are supported and their
        held-out MAE. For solvent MIXTURES use
        ``umapka.mixtures.predict_mixed_solvent_pka`` instead - a single
        `solvent` string here always means one pure solvent.

        `salt` / `salt_concentration` (mol/L) apply a physics-based
        ionic-strength correction (see ``umapka.solvation``) on top of
        the base prediction for whichever solvent you picked; they
        don't change which solvent the base prediction is for. Use
        ``predict_detailed`` if you want the correction tier/warnings
        rather than just the final float.
        """
        return self.predict_detailed(smiles, solvent, salt, salt_concentration)["pKa"]

    def predict_detailed(self, smiles: str, solvent: str = "water",
                          salt: str | None = None,
                          salt_concentration: float | None = None) -> dict:
        """Like ``predict``, but returns
        {"pKa": float, "base_pKa": float, "solvent": str, "correction": dict}
        where ``correction`` is the raw dict from
        ``solvation.predict_salt_correction`` - includes which tier
        fired (davies / davies+ion-pairing / pitzer / none) and any
        validity warnings. Inspect this rather than trusting the
        headline number blindly when a salt is specified.
        """
        from . import solvents as _solvents
        info = _solvents.resolve_solvent(solvent)
        prot, deprot = protonation_pair(smiles)
        base = self._base_pka(self.features(prot, deprot), solvent)
        if salt is None:
            return {"pKa": base, "base_pKa": base, "solvent": info.name,
                    "correction": {"shift": 0.0, "tier": "none",
                                   "note": "no salt specified"}}
        if info.solvation_key is None:
            raise ValueError(
                f"no ionic-strength model available for {info.name} - "
                f"umapka.solvation only covers "
                f"{sorted(s.solvation_key for s in _solvents.SOLVENTS.values() if s.solvation_key)}"
            )

        import importlib
        solvation = importlib.import_module(".solvation", __package__)
        # the site actually used by protonation_pair() is, by
        # construction, sites(smiles)[0] - same priority-ordered scan
        site_kind = self.sites(smiles)[0]["kind"]
        correction = solvation.predict_salt_correction(
            site_kind, salt, salt_concentration, solvent=info.solvation_key)
        return {"pKa": base + correction["shift"], "base_pKa": base,
                "solvent": info.name, "correction": correction}

    def sites(self, smiles: str) -> list[dict]:
        """List titratable sites, each with an index for ``predict_site``."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles}")
        mol = neutralize(mol)
        found, seen = [], set()
        for kind, table in (("acid", ACID_SITES), ("base", BASE_SITES)):
            for group, smarts, ai in table:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    continue
                for match in mol.GetSubstructMatches(patt):
                    idx = match[ai]
                    if idx in seen:
                        continue
                    seen.add(idx)
                    found.append({
                        "index": len(found), "atom": idx, "group": group,
                        "kind": kind,
                        "element": mol.GetAtomWithIdx(idx).GetSymbol(),
                    })
        return found

    def predict_site(self, smiles: str, site_index: int,
                      solvent: str = "water",
                      salt: str | None = None,
                      salt_concentration: float | None = None) -> float:
        """Predict pKa for one chosen site, other sites left neutral,
        in `solvent` (see ``predict`` for supported solvents and what
        ``salt``/``salt_concentration`` do).
        """
        from . import solvents as _solvents
        info = _solvents.resolve_solvent(solvent)
        mol = neutralize(Chem.MolFromSmiles(smiles))
        sites = self.sites(smiles)
        if site_index >= len(sites):
            raise IndexError(f"molecule has only {len(sites)} sites")
        site = sites[site_index]
        if site["kind"] == "acid":
            other = _shift_hydrogen(mol, site["atom"], -1, -1)
            pair = (mol, other)
        else:
            other = _shift_hydrogen(mol, site["atom"], +1, +1)
            pair = (other, mol)
        if pair[0] is None or pair[1] is None:
            raise RuntimeError("could not construct the protonation pair")
        pair_feat = self.features(Chem.MolToSmiles(pair[0]), Chem.MolToSmiles(pair[1]))
        base = self._base_pka(pair_feat, solvent)
        if salt is None:
            return base
        if info.solvation_key is None:
            raise ValueError(
                f"no ionic-strength model available for {info.name} - see "
                f"umapka.solvents.SOLVENTS for which solvents support it"
            )
        import importlib
        solvation = importlib.import_module(".solvation", __package__)
        correction = solvation.predict_salt_correction(
            site["kind"], salt, salt_concentration, solvent=info.solvation_key)
        return base + correction["shift"]

    def predict_all_sites(self, smiles: str, solvent: str = "water",
                           salt: str | None = None,
                           salt_concentration: float | None = None) -> list[dict]:
        """Score EVERY detected titratable site, not just the first one
        found by SMARTS priority order (which is what ``predict()`` uses).

        Borrowed from pKalculator's strategy: since umapka's regressor
        can already evaluate any single site via ``predict_site()``, and
        ``sites()`` already enumerates every candidate, there's no reason
        to only ever look at the first match. Useful for molecules with
        multiple plausible acid/base sites where you want to know which
        one is actually most acidic/basic, not just which one the
        priority table happened to find first.

        Returns a list of site dicts (same shape as ``sites()``) each
        with an added "pKa" key, sorted by pKa ascending. Note this does
        NOT solve sequential multi-deprotonation (that's a different,
        harder problem - see ``predict_detailed`` discussion and the
        free-energy-difference approach); this only ranks INDEPENDENT
        single-site predictions for one fixed molecule, each computed
        with all other sites left neutral.
        """
        sites = self.sites(smiles)
        results = []
        for site in sites:
            try:
                pka = self.predict_site(smiles, site["index"], solvent=solvent,
                                         salt=salt,
                                         salt_concentration=salt_concentration)
                results.append({**site, "pKa": pka})
            except Exception as e:
                results.append({**site, "pKa": None, "error": str(e)})
        scored = [r for r in results if r["pKa"] is not None]
        unscored = [r for r in results if r["pKa"] is None]
        return sorted(scored, key=lambda r: r["pKa"]) + unscored

    _PROTONATED_BASE_PATTERN = Chem.MolFromSmarts("[#7+;H1,H2,H3]")

    def _sites_on_current_mol(self, mol) -> list[tuple[str, int]]:
        """Like sites(), but works DIRECTLY on the given mol object
        without calling neutralize() first, AND includes protonated
        base sites (ammonium-type: N+ with a remaining H) as candidate
        removable-proton sites, not just ACID_SITES.

        Why: once a base site (amine, guanidine, pyridine, ...) has
        been protonated - either by _build_fully_protonated() at the
        start of a chain, or in principle at any point - it becomes,
        chemically, just another site that can lose a proton. There is
        no meaningful difference at that point between "a carboxylic
        acid losing H+" and "an ammonium losing H+"; both are acid
        dissociations. The generic pattern [#7+;H1,H2,H3] catches
        protonated primary/secondary/tertiary amines, anilinium,
        pyridinium, protonated guanidinium/amidinium, etc. without
        needing to track which specific atom got protonated earlier.

        Also does NOT call neutralize() first, for the same reason as
        before: after removing one proton, the resulting anion (or,
        now, after removing a proton from an ammonium, the resulting
        neutral amine) has a real, intentional charge state that
        neutralize()'s cleanup SMARTS would otherwise undo.
        """
        found = []
        for group, smarts, ai in ACID_SITES:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            for match in mol.GetSubstructMatches(patt):
                found.append((group, match[ai]))
        if self._PROTONATED_BASE_PATTERN is not None:
            for match in mol.GetSubstructMatches(self._PROTONATED_BASE_PATTERN):
                found.append(("protonated_base", match[0]))
        return found

    def _build_fully_protonated(self, mol):
        """From a NEUTRAL mol (as neutralize() returns), protonate every
        detected BASE_SITES match to reach the fully-protonated
        macrostate - the correct starting point for a complete
        deprotonation chain covering both acid AND base sites.

        Acid sites are already at their fully-protonated H-count by
        convention (neutralize() gives COOH, not COO-, an amine as NH2
        not NH3+), so only base sites need this extra step. Without it,
        a molecule's own amine/guanidine/etc. ionization would be
        silently skipped entirely - which is exactly the gap found when
        testing this method on amino-diacid molecules.
        """
        current = mol
        for _ in range(10):  # generous cap; molecules rarely have >10 base sites
            found_idx = None
            for group, smarts, ai in BASE_SITES:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    continue
                matches = current.GetSubstructMatches(patt)
                if matches:
                    found_idx = matches[0][ai]
                    break
            if found_idx is None:
                break
            protonated = _shift_hydrogen(current, found_idx, +1, +1)
            if protonated is None:
                break
            current = protonated
        return current

    def _predict_free_energy(self, feat: np.ndarray) -> float:
        """Predict pKa from a feature row, transparently supporting
        either a single regressor or a list of regressors (ensemble -
        averages predictions). free_energy_ensemble.pkl (validated:
        1.454 MAE, 96.1%/90.5% ordering for 2/3-step chains) is a list
        of 5 models; older single-model files still work unchanged.
        """
        model = self._free_energy_model
        if isinstance(model, (list, tuple)):
            return float(np.mean([float(m.predict(feat)[0]) for m in model]))
        return float(model.predict(feat)[0])

    def predict_chain(self, smiles: str, max_steps: int = 4) -> dict:
        """Predict a genuine SEQUENCE of deprotonations across BOTH acid
        and base sites - find the most acidic removable proton, remove
        it, find the next most acidic on the resulting structure, and
        so on. Starts from the fully-protonated macrostate (every
        detected base site protonated, e.g. an amine as NH3+), not the
        plain neutral molecule - otherwise a molecule's own amine/
        guanidine/etc. ionization would be silently skipped, which is
        exactly the gap found when this method was first tested on
        amino-diacid molecules (it only walked ACID_SITES).

        This is a different problem from predict_all_sites(), which
        ranks independent single-site predictions on the SAME starting
        molecule with every other site left neutral; predict_chain()
        actually walks the ionization path.

        Requires a free-energy-difference model loaded via
        PkaPredictor(..., free_energy_model_path=...) - see
        train_free_energy_model.py. Raises if none was loaded.

        IMPORTANT - read before trusting results:
          - This code has been run-tested end-to-end (base version) and
            the underlying model has been validated with a tuned,
            feature-enriched, 5-model ensemble (free_energy_ensemble.pkl):
            scaffold-split MAE 1.454, single-site subset 1.347. Don't
            use this for single-site molecules; use predict() instead
            (0.994 MAE, still meaningfully better for that case).
          - Ordering accuracy on held-out chains: 96.1% for 2-step
            (173/180), 90.5% for 3-step (19/21) - a large improvement
            over earlier versions (88.9%/61.9% baseline,
            83.9%/61.9% with a labeling bug found and fixed along the
            way). Achieved via hyperparameter tuning, an explicit
            step-number feature, and 5-model ensembling - NOT via a
            more complex architecture, which was tried and made things
            worse (a small neural net scored 1.632 MAE, 78.3%/66.7% -
            classic small-data regime where gradient-boosted trees
            beat neural nets).
          - Known failure modes, from direct inspection of held-out
            mis-ordered chains: near-degenerate true pKa gaps (<0.5
            apart - often not meaningful orderings to begin with),
            triprotic amino-diacids, and fused heteroaromatic systems
            with coupled tautomers.
          - This method does NOT sort the output to force monotonic
            ordering. Check the "monotonic" key - if False, treat the
            chain's ordering as unreliable rather than trusting it.
          - The starting state is the fully-protonated macrostate,
            not plain neutralize() output - the FIRST entry in
            "states" may therefore carry a positive charge (e.g. an
            amino acid's ammonium/diacid form), which is intentional.
          - No phosphonic/phosphoric acid site coverage - molecules
            with only this chemistry will return an empty chain.

        Returns {"states": [SMILES, ...], "pKas": [float, ...],
                 "monotonic": bool, "warning": str | None}
        """
        if self._free_energy_model is None:
            raise RuntimeError(
                "predict_chain() needs a free-energy-difference model. "
                "Construct PkaPredictor(..., free_energy_model_path="
                "'free_energy_model_scaffold.pkl') (or similar) - this "
                "is a SEPARATE file from the main model_core.pkl."
            )

        mol = neutralize(Chem.MolFromSmiles(smiles))
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles}")
        mol = self._build_fully_protonated(mol)

        states_smiles = [Chem.MolToSmiles(mol)]
        pkas = []
        current_mol = mol
        # single-state embedding of the starting structure
        h_current = self.pool(self.embeddings(_smiles_to_atoms(Chem.MolToSmiles(current_mol))))

        for _ in range(max_steps):
            candidates = self._sites_on_current_mol(current_mol)
            if not candidates:
                break

            step_number = len(pkas) + 1  # 1-indexed, matches training exactly
            best = None  # (pka, deprot_mol, h_deprot)
            for group, atom_idx in candidates:
                deprot = _shift_hydrogen(current_mol, atom_idx, -1, -1)
                if deprot is None:
                    continue
                h_deprot = self.pool(self.embeddings(
                    _smiles_to_atoms(Chem.MolToSmiles(deprot))))
                # SAME feature convention as train_tuned_ensemble.py:
                # [h_before, h_after, h_after - h_before, step_number] -
                # note this is the OPPOSITE order from features()'s
                # [h_prot, h_deprot, h_prot-h_deprot]. Getting this
                # backwards would silently feed the model mis-signed
                # input.
                feat = np.concatenate(
                    [h_current, h_deprot, h_deprot - h_current, [step_number]]
                ).reshape(1, -1)
                pka = self._predict_free_energy(feat)
                if best is None or pka < best[0]:
                    best = (pka, deprot, h_deprot)

            if best is None:
                break
            pka, current_mol, h_current = best
            pkas.append(pka)
            states_smiles.append(Chem.MolToSmiles(current_mol))

        monotonic = all(pkas[i] <= pkas[i + 1] for i in range(len(pkas) - 1))
        return {
            "states": states_smiles,
            "pKas": pkas,
            "monotonic": monotonic,
            "warning": None if monotonic else (
                "predicted pKa sequence is not monotonically increasing "
                "- this chain may involve a hard chemotype (see "
                "predict_chain docstring); do not trust the ordering"
            ),
        }
