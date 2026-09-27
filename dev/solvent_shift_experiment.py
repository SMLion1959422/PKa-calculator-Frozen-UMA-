"""Solvent-shift experiment.

Hypothesis: predicting the solvent-induced SHIFT from water, with
physical solvent descriptors and a site-focused solute representation,
generalizes to unseen solvents better than predicting absolute pKa per
solvent.

Four models (plus one control), identical splits and rows:

  M1   absolute   pKa_S = f(solute, eps/78.4, protic)       current formulation
  M1b  absolute   pKa_S = f(solute, solvent descriptors)    control: descriptors, no shift
  M2   offset     pKa_S = pKa_W + Delta_S                   one number per solvent (per charge type)
  M3   descriptors pKa_S = pKa_W + g(d_S, pKa_W)            linear in descriptors
  M4   proposed   pKa_S = M3 + h(solute, d_S, pKa_W)        + site-focused solute features

The key comparison is M2 vs M4.

Water anchor pKa_W, reported both ways:
  measured    only rows whose reaction also has a measured water pKa
  predicted   all rows; pKa_W from a water model trained inside each fold
              (the test fold's reactions are excluded from it)

Splits:
  random      5-fold, grouped by reaction (a solute never spans folds)
  LOSO        leave-one-solvent-out, every non-water solvent (centrepiece)
  LOFO        leave-one-chemical-family-out (site type of the acid)
  published   the dataset's own train/test split, if --split-dir is given

Calibration (LOSO only): the held-out solvent gets k = 0, 1, 2, 5, 10
measured pKas; each model's predictions are shifted by the mean
residual on those k points and scored on the remaining molecules
(the same evaluation set for every k within a draw; 20 draws).

Solute features: RDKit site features (umapka.rdkit_site; any machine) or
site-focused UMA features (--features uma; GPU + facebook/UMA access),
both rooted at the atom that loses the proton, taken from the reaction.

Data: Nevolianis et al. D2A-pKa (Zenodo 15604045, CC BY 4.0), columns
reaction_smiles ("HA>>A-"), solvent_smiles, pKa_avg.

    python dev/solvent_shift_experiment.py zenodo [--features uma --cache C.pkl]
        # "zenodo" downloads record 15604045 into ./anion_data (once) and
        # finds D2A-pKa.csv and the published split files itself
    python dev/solvent_shift_experiment.py path/to/D2A-pKa.csv --split-dir path/to/data_splits
"""
import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from umapka.sites import Site, find_sites, CHARGE_SHARE
from umapka.rdkit_site import pair_features
from umapka.solvent_descriptors import SMILES_TO_NAME, descriptor_vector, DESCRIPTORS

RDLogger.DisableLog("rdApp.*")
WATER = "Water"
PROTIC = {"Water": 1.0, "Methanol": 0.5, "Ethanol": 0.5, "EthyleneGlycol": 0.5}
KS = (0, 1, 2, 5, 10)
N_DRAWS = 20


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------
def _canon(m):
    return Chem.MolToSmiles(m, isomericSmiles=False)


def locate_site(prot_smi, deprot_smi):
    """Index (in the protonated mol) of the atom that loses the proton.
    Brute force: put the proton back on each deprotonated atom in turn
    and compare with the protonated species."""
    prot = Chem.MolFromSmiles(prot_smi)
    dep = Chem.MolFromSmiles(deprot_smi)
    if prot is None or dep is None:
        return None, None
    target = _canon(prot)
    for a in dep.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        rw = Chem.RWMol(dep)
        at = rw.GetAtomWithIdx(a.GetIdx())
        at.SetFormalCharge(at.GetFormalCharge() + 1)
        at.SetNumExplicitHs(at.GetTotalNumHs() + 1)
        at.SetNoImplicit(True)
        try:
            m = rw.GetMol()
            Chem.SanitizeMol(m)
        except Exception:
            continue
        if _canon(m) != target:
            continue
        match = prot.GetSubstructMatch(m)
        if match:
            return prot, match[a.GetIdx()]
    return prot, None


def family(prot, idx):
    at = prot.GetAtomWithIdx(idx)
    el = at.GetSymbol()
    if at.GetFormalCharge() > 0:
        return "cationic acid (BH+)"
    if el == "C":
        return "C-H acid"
    if el == "S":
        return "S-H acid"
    if el == "N":
        return "N-H acid"
    if el == "O":
        if prot.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]")) and any(
                idx in m for m in prot.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))):
            return "carboxylic acid"
        if any(n.GetIsAromatic() for n in at.GetNeighbors()):
            return "phenol"
        return "other O-H acid"
    return "other"


def site_object(prot, idx):
    """Site for feature extraction, with charge sharing when the SMARTS
    table knows the group."""
    for s in find_sites(prot):
        if s.atom == idx:
            return Site(s.index, idx, s.group, "acid", s.element, s.charge_atoms)
    return Site(-1, idx, "reaction_site", "acid", prot.GetAtomWithIdx(idx).GetSymbol(),
                [(idx, 1.0)])


def load(path, max_rows=None):
    df = pd.read_csv(path)
    rows, bad = [], 0
    for r in df.itertuples():
        name = SMILES_TO_NAME.get(str(r.solvent_smiles))
        if name is None or not (-15 < r.pKa_avg < 45):
            continue
        try:
            a, b = str(r.reaction_smiles).split(">>")
        except ValueError:
            bad += 1
            continue
        rows.append((a.strip(), b.strip(), name, float(r.pKa_avg)))
    d = pd.DataFrame(rows, columns=["prot", "deprot", "solvent", "pka"])
    d = d.groupby(["prot", "deprot", "solvent"], as_index=False)["pka"].mean()
    d["rxn"] = d.prot + ">>" + d.deprot
    info = {}
    for rxn, prot_s, dep_s in d[["rxn", "prot", "deprot"]].drop_duplicates().itertuples(index=False):
        prot, idx = locate_site(prot_s, dep_s)
        info[rxn] = (prot, idx)
    ok = d.rxn.map(lambda x: info[x][1] is not None)
    msg = (f"rows {len(d)}; site located for {ok.sum()} ({(~ok).sum()} dropped: "
           f"reaction not a single-proton transfer on the same skeleton)")
    print(msg)
    load.note = msg
    d = d[ok].reset_index(drop=True)
    d["family"] = d.rxn.map(lambda x: family(*info[x]))
    d["charge_type"] = d.family.map(lambda f: "cationic" if f.startswith("cationic") else "neutral")
    if max_rows:
        d = d.sample(min(max_rows, len(d)), random_state=0).reset_index(drop=True)
    return d, info


# ---------------------------------------------------------------------
# features
# ---------------------------------------------------------------------
def solute_features(d, info, mode, cache_path=None, model_path="models/model_core_v2.pkl"):
    rxns = d.rxn.unique()
    if mode == "rdkit":
        return {r: pair_features(info[r][0], info[r][1], "acid") for r in rxns}
    import joblib
    from umapka import PkaPredictor
    pred = PkaPredictor(model_path, multisolvent_model_path=None)
    cache = joblib.load(cache_path) if cache_path and os.path.exists(cache_path) else {}
    new = 0
    for r in rxns:
        if cache.get(r) is not None:
            continue
        prot, idx = info[r]
        try:
            cache[r] = pred.features_site(prot, site_object(prot, idx))[0]
        except Exception as e:
            cache[r] = None
            print(f"  skip {r}: {e}")
        new += 1
        if new % 200 == 0 and cache_path:
            joblib.dump(cache, cache_path + ".tmp")
            os.replace(cache_path + ".tmp", cache_path)
            print(f"  checkpoint: {sum(v is not None for v in cache.values())} reactions featurized", flush=True)
    if cache_path:
        joblib.dump(cache, cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)
    return {r: cache[r] for r in rxns if cache.get(r) is not None}


# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------
def lgbm(n=600):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=31,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.4, verbose=-1, random_state=0)


def solvent_block(names, kind):
    if kind == "flag":
        return np.array([[DESCRIPTORS[s][0] / 78.4, PROTIC.get(s, 0.0)] for s in names])
    return np.array([descriptor_vector(s) for s in names])


def shift_design(D, pw, ct):
    """M3 design: descriptors, descriptors x pKa_W, per charge type."""
    c = (ct == "cationic").astype(float)[:, None]
    base = np.hstack([D, D * pw[:, None], pw[:, None]])
    return np.hstack([base, base * c, c])


class Water:
    """In-fold aqueous model on water rows."""
    def __init__(self, F, train_water):
        X = np.array([F[r] for r in train_water.rxn])
        self.m = lgbm(800).fit(X, train_water.pka.values)

    def __call__(self, rxns, F):
        return self.m.predict(np.array([F[r] for r in rxns]))


def fit_predict(train, test, F, anchor):
    """Return {model: predictions on test (non-water rows)}."""
    tr_w = train[train.solvent == WATER]
    tr = train[train.solvent != WATER]
    te = test[test.solvent != WATER]
    if anchor == "measured":
        wmap = pd.concat([train, test])
        wmap = wmap[wmap.solvent == WATER].set_index("rxn").pka.to_dict()
        tr = tr[tr.rxn.isin(wmap)]
        te = te[te.rxn.isin(wmap)]
        pw_tr = tr.rxn.map(wmap).values
        pw_te = te.rxn.map(wmap).values
    else:
        test_rxns = set(test.rxn)
        tw = tr_w[~tr_w.rxn.isin(test_rxns)]
        water = Water(F, tw)
        pw_te = water(te.rxn, F)
        # training rows get OUT-OF-FOLD water predictions, so the offsets
        # and shift models see the same kind of water error as the test rows
        pw_tr = np.empty(len(tr))
        rx = tr.rxn.values
        folds = GroupKFold(5).split(rx, groups=rx)
        for a, b in folds:
            held = set(rx[b])
            wm = Water(F, tw[~tw.rxn.isin(held)])
            pw_tr[b] = wm(rx[b], F)
    if len(te) == 0 or len(tr) < 20:
        return te, {}
    Xs_tr = np.array([F[r] for r in tr.rxn])
    Xs_te = np.array([F[r] for r in te.rxn])
    ct_tr, ct_te = tr.charge_type.values, te.charge_type.values
    out = {}

    # M1 / M1b: absolute pKa, trained on all training rows including water
    full = train
    Xf = np.array([F[r] for r in full.rxn])
    for name, kind in (("M1 absolute (eps, protic)", "flag"),
                       ("M1b absolute (descriptors)", "desc")):
        m = lgbm().fit(np.hstack([Xf, solvent_block(full.solvent, kind)]), full.pka.values)
        out[name] = m.predict(np.hstack([Xs_te, solvent_block(te.solvent, kind)]))

    # M2: water + offset per (solvent, charge type); unseen -> global mean
    shift = tr.pka.values - pw_tr
    key = list(zip(tr.solvent, ct_tr))
    off = pd.Series(shift, index=pd.MultiIndex.from_tuples(key)).groupby(level=[0, 1]).median()
    glob = {c: np.median(shift[ct_tr == c]) if (ct_tr == c).any() else np.median(shift)
            for c in ("neutral", "cationic")}
    out["M2 water + offset"] = pw_te + np.array(
        [off.get((s, c), glob[c]) for s, c in zip(te.solvent, ct_te)])

    # M3: water + linear descriptor model of the shift
    D_tr, D_te = solvent_block(tr.solvent, "desc"), solvent_block(te.solvent, "desc")
    r3 = Ridge(alpha=1.0).fit(shift_design(D_tr, pw_tr, ct_tr), shift)
    s3_tr = r3.predict(shift_design(D_tr, pw_tr, ct_tr))
    out["M3 water + descriptors"] = pw_te + r3.predict(shift_design(D_te, pw_te, ct_te))

    # M4: M3 + solute-specific residual shift
    Z_tr = np.hstack([Xs_tr, D_tr, pw_tr[:, None], (ct_tr == "cationic")[:, None]])
    Z_te = np.hstack([Xs_te, D_te, pw_te[:, None], (ct_te == "cationic")[:, None]])
    m4 = lgbm().fit(Z_tr, shift - s3_tr)
    out["M4 water + descriptors + site features"] = out["M3 water + descriptors"] + m4.predict(Z_te)
    out["_water_anchor_only"] = pw_te      # reference: pretend pKa_S = pKa_W
    return te, out


# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------
def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def calibration_curve(y, preds, rng):
    """{model: {k: mean MAE over draws}} with a fixed eval set per draw."""
    n = len(y)
    if n < max(KS) + 10:
        return None
    res = {m: {k: [] for k in KS} for m in preds}
    for _ in range(N_DRAWS):
        perm = rng.permutation(n)
        pool, ev = perm[:max(KS)], perm[max(KS):]
        for m, p in preds.items():
            for k in KS:
                bias = np.mean(y[pool[:k]] - p[pool[:k]]) if k else 0.0
                res[m][k].append(mae(y[ev], p[ev] + bias))
    return {m: {k: float(np.mean(v)) for k, v in ks.items()} for m, ks in res.items()}


ZENODO_RECORD = "15604045"


def resolve_data(arg, split_dir, dest="anion_data"):
    """Return (csv path, split dir). ``arg`` may be a CSV path, a directory
    to search, or "zenodo" to download the record first."""
    import glob
    if arg == "zenodo":
        if not glob.glob(f"{dest}/**/D2A-pKa.csv", recursive=True):
            import io, zipfile, requests
            rec = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60).json()
            os.makedirs(dest, exist_ok=True)
            for f in rec["files"]:
                key, url = f["key"], f["links"]["self"]
                print(f"downloading {key} ({f['size'] / 1e6:.0f} MB)", flush=True)
                blob = requests.get(url, timeout=600).content
                if key.endswith(".zip"):
                    zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
                else:
                    open(os.path.join(dest, key), "wb").write(blob)
        arg = dest
    if os.path.isdir(arg):
        hits = glob.glob(f"{arg}/**/D2A-pKa.csv", recursive=True)
        if not hits:
            raise FileNotFoundError(f"no D2A-pKa.csv under {arg}; files there: "
                                    f"{glob.glob(f'{arg}/**/*.csv', recursive=True)[:20]}")
        arg = hits[0]
    if not os.path.isfile(arg):
        raise FileNotFoundError(f"data file not found: {arg!r}")
    if split_dir is None:
        tr = glob.glob(os.path.join(os.path.dirname(arg), "**", "D2A-pKa-train.csv"), recursive=True)
        split_dir = os.path.dirname(tr[0]) if tr else None
    print(f"data: {arg}\nsplits: {split_dir}")
    return arg, split_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help='CSV path, a directory containing it, or "zenodo"')
    ap.add_argument("--split-dir")
    ap.add_argument("--features", choices=("rdkit", "uma"), default="rdkit")
    ap.add_argument("--cache", default="solvent_shift_uma_cache.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="subsample (debugging)")
    args = ap.parse_args()
    out_dir = args.out or f"results/solvent_shift_{args.features}"
    os.makedirs(out_dir, exist_ok=True)

    args.data, args.split_dir = resolve_data(args.data, args.split_dir)
    d, info = load(args.data, args.max_rows)
    F = solute_features(d, info, args.features, args.cache)
    d = d[d.rxn.isin(F)].reset_index(drop=True)
    counts = d.groupby("solvent").size().sort_values(ascending=False)
    n_meas = d[(d.solvent != WATER) & d.rxn.isin(set(d[d.solvent == WATER].rxn))].shape[0]
    lines = [load.note,
             f"features: {args.features}; rows {len(d)}; reactions {d.rxn.nunique()}",
             "rows per solvent: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
             f"non-water rows with a measured water pKa for the same reaction: {n_meas}",
             "families (non-water rows): " + ", ".join(
                 f"{k} {v}" for k, v in d[d.solvent != WATER].family.value_counts().items()), ""]
    records, calib = [], []
    rng = np.random.default_rng(0)

    def run(split, fold, train, test, anchor):
        te, preds = fit_predict(train, test, F, anchor)
        if not preds:
            return
        y = te.pka.values
        for m, p in preds.items():
            records.append(dict(split=split, fold=fold, anchor=anchor, model=m,
                                n=len(y), mae=mae(y, p)))
        if split == "LOSO":
            cc = calibration_curve(y, {m: p for m, p in preds.items()}, rng)
            if cc:
                for m, ks in cc.items():
                    for k, v in ks.items():
                        calib.append(dict(solvent=fold, anchor=anchor, model=m, k=k, mae=v))

    for anchor in ("measured", "predicted"):
        g = GroupKFold(5)
        for i, (a, b) in enumerate(g.split(d, groups=d.rxn)):
            run("random (solute-grouped)", i, d.iloc[a], d.iloc[b], anchor)
        for s in [s for s in counts.index if s != WATER and counts[s] >= 20]:
            run("LOSO", s, d[d.solvent != s], d[d.solvent == s], anchor)
        fams = d[d.solvent != WATER].family.value_counts()
        for f in [f for f, n in fams.items() if n >= 30]:
            run("LOFO", f, d[d.family != f], d[d.family == f], anchor)
        if args.split_dir:
            tr_r = set(load(os.path.join(args.split_dir, "D2A-pKa-train.csv"))[0].rxn)
            te_r = set(load(os.path.join(args.split_dir, "D2A-pKa-test.csv"))[0].rxn)
            run("published", "test", d[d.rxn.isin(tr_r)], d[d.rxn.isin(te_r)], anchor)

    R = pd.DataFrame(records)
    R.to_csv(f"{out_dir}/per_fold.csv", index=False)
    C = pd.DataFrame(calib)
    if len(C):
        C.to_csv(f"{out_dir}/calibration.csv", index=False)

    # row-weighted MAE per split, anchor, model
    R["abs_sum"] = R.mae * R.n
    agg = R.groupby(["split", "anchor", "model"]).agg(n=("n", "sum"), s=("abs_sum", "sum"))
    agg["MAE"] = agg.s / agg.n
    for (split, anchor), grp in agg.groupby(level=[0, 1]):
        lines.append(f"== {split} | water anchor: {anchor} ==")
        for (_, _, model), r in grp.iterrows():
            lines.append(f"  {model:42s} MAE {r.MAE:6.2f}  (n={int(r.n)})")
        lines.append("")
    lines.append("== LOSO per solvent (MAE) ==")
    piv = R[R.split == "LOSO"].pivot_table(index=["anchor", "fold"], columns="model", values="mae")
    lines.append(piv.round(2).to_string())
    lines.append("")
    if len(C):
        lines.append("== calibration: MAE vs k measured points in the unseen solvent "
                     "(mean over LOSO solvents) ==")
        cp = C.groupby(["anchor", "model", "k"]).mae.mean().unstack("k")
        lines.append(cp.round(2).to_string())
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for anchor in C.anchor.unique():
                fig, ax = plt.subplots(figsize=(6, 4))
                for m, row in cp.loc[anchor].iterrows():
                    if m.startswith("_"):
                        continue
                    ax.plot(list(row.index), row.values, marker="o", label=m)
                ax.set_xlabel("measured pKa values in the unseen solvent (k)")
                ax.set_ylabel("MAE on the remaining molecules (pKa units)")
                ax.set_title(f"Leave-one-solvent-out calibration ({args.features}, {anchor} water pKa)")
                ax.set_xticks(list(KS))
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(f"{out_dir}/calibration_{anchor}.png", dpi=150)
                plt.close(fig)
        except ImportError:
            pass
    text = "\n".join(lines)
    print(text)
    open(f"{out_dir}/summary.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
