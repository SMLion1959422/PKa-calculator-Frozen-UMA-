"""UMA-free ablation: how much do (1) correct site assignment and
(2) site-centred instead of whole-molecule features matter?

Uses a cheap RDKit featurizer so it runs anywhere in ~1 minute; the
point is the *difference* between arms, which is what motivates the
same two changes to the UMA pipeline (see METHODOLOGY.md). It does not
say anything about absolute UMA accuracy.

Arms (same LightGBM, same train/test molecules):
  A  global Morgan counts, site chosen by SMARTS priority (what
     umapka.protonation_pair does today); "no site" rows fall back to
     global features only
  B  global Morgan counts + site-centred features at the SMARTS site
  C  global Morgan counts + site-centred features at the ANNOTATED site
     (marvin_atom in the Baltruschat & Czodrowski SDFs)
  D  site-centred features only, annotated site

Data: github.com/czodrowskilab/Machine-learning-meets-pKa (CC BY 4.0).
Clone it and pass the datasets dir:
    python dev/site_ablation.py path/to/Machine-learning-meets-pKa/datasets
"""
import sys
import numpy as np
import lightgbm as lgb
from rdkit import Chem, RDLogger
from umapka.sites import find_sites, neutralize
from umapka.rdkit_site import global_features, site_features

RDLogger.DisableLog("rdApp.*")
D = sys.argv[1] if len(sys.argv) > 1 else "mlpka/datasets"


def glob(mol):
    return global_features(mol)


def site_feats(mol, atom, kind):
    """Site-centred features (umapka.rdkit_site); kind "acidic"/"basic"."""
    return site_features(mol, atom, "acid" if kind == "acidic" else "base")


def load(name):
    rows = []
    for m in Chem.SDMolSupplier(f"{D}/{name}.sdf"):
        if m is None:
            continue
        p = m.GetPropsAsDict()
        n = neutralize(m)
        sites = find_sites(n)
        smarts_atom = sites[0].atom if sites else None
        kind_s = ("acidic" if sites and sites[0].kind == "acid" else "basic")
        rows.append(dict(y=p["pKa"], g=glob(n),
                         s_smarts=site_feats(n, smarts_atom, kind_s) if sites else None,
                         s_true=site_feats(n, p["marvin_atom"], p["marvin_pKa_type"]),
                         agree=bool(sites) and smarts_atom == p["marvin_atom"]))
    return rows


def arm(rows, which):
    X = []
    zeros = np.zeros_like(rows[0]["s_true"])
    for r in rows:
        if which == "A":
            X.append(r["g"])
        elif which == "B":
            X.append(np.concatenate([r["g"], r["s_smarts"] if r["s_smarts"] is not None else zeros]))
        elif which == "C":
            X.append(np.concatenate([r["g"], r["s_true"]]))
        else:
            X.append(r["s_true"])
    return np.array(X), np.array([r["y"] for r in rows])


def agreement(name):
    """How often SMARTS priority (protonation_pair) picks the annotated atom."""
    c = {"same": 0, "found_not_first": 0, "not_covered": 0, "no_site": 0}
    for m in Chem.SDMolSupplier(f"{D}/{name}.sdf"):
        if m is None:
            continue
        a = m.GetPropsAsDict()["marvin_atom"]
        sites = find_sites(neutralize(m))
        if not sites:
            c["no_site"] += 1
        elif sites[0].atom == a:
            c["same"] += 1
        elif a in [s.atom for s in sites]:
            c["found_not_first"] += 1
        else:
            c["not_covered"] += 1
    n = sum(c.values())
    return n, {k: f"{v / n:.1%}" for k, v in c.items()}


for name in ("combined_training_datasets_unique",
             "novartis_cleaned_mono_unique_notraindata",
             "AvLiLuMoVe_cleaned_mono_unique_notraindata"):
    print("site agreement", name, *agreement(name))
print()

train = load("combined_training_datasets_unique")
tests = {n: load(f"{n}_cleaned_mono_unique_notraindata") for n in ("novartis", "AvLiLuMoVe")}
print(f"train n={len(train)}")
print(f"{'arm':44s}" + "".join(f"{n:>12s}" for n in tests) + f"{'Nov: agree':>12s}{'disagree':>10s}")
labels = {"A": "A global only (SMARTS site irrelevant)",
          "B": "B global + site feats @ SMARTS site",
          "C": "C global + site feats @ annotated site",
          "D": "D site feats only @ annotated site"}
for w in "ABCD":
    X, y = arm(train, w)
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                          verbose=-1, random_state=0).fit(X, y)
    line = f"{labels[w]:44s}"
    for n, rows in tests.items():
        Xt, yt = arm(rows, w)
        err = np.abs(m.predict(Xt) - yt)
        line += f"{err.mean():12.3f}"
        if n == "novartis":
            ag = np.array([r["agree"] for r in rows])
            nov = (err[ag].mean(), err[~ag].mean())
    print(line + f"{nov[0]:12.3f}{nov[1]:10.3f}")


# ---------------------------------------------------------------------
# Arm E: fully automatic. Model C scores EVERY detected site; the site
# is then chosen without the annotation. Rules compared:
#   E1 SMARTS priority (acids first)             = today's behaviour
#   E2 most acidic acid if any acid predicted < 12, else most basic base
#   E3 predicted pKa closest to the 2-12 window centre among in-window
#      sites (the datasets only contain pKas measured in that window)
# ---------------------------------------------------------------------
print("\nautomatic site choice with model C (site feats scored at every SMARTS site):")
Xc, yc = arm(train, "C")
mc = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                       verbose=-1, random_state=0).fit(Xc, yc)


def all_site_preds(name):
    out = []
    for m in Chem.SDMolSupplier(f"{D}/{name}_cleaned_mono_unique_notraindata.sdf"):
        if m is None:
            continue
        p = m.GetPropsAsDict()
        n = neutralize(m)
        g = glob(n)
        cands = []
        for s in find_sites(n):
            kind = "acidic" if s.kind == "acid" else "basic"
            f = np.concatenate([g, site_feats(n, s.atom, kind)])
            cands.append((s, float(mc.predict(f.reshape(1, -1))[0])))
        out.append((p["pKa"], cands))
    return out


def choose(cands, rule):
    if not cands:
        return None
    if rule == "E1":
        return cands[0][1]
    if rule == "E2":
        acids = [v for s, v in cands if s.kind == "acid" and v < 12]
        if acids:
            return min(acids)
        bases = [v for s, v in cands if s.kind == "base"]
        return max(bases) if bases else cands[0][1]
    inwin = [v for _, v in cands if 2 <= v <= 12]
    pool = inwin or [v for _, v in cands]
    return min(pool, key=lambda v: abs(v - 7))


for name in ("novartis", "AvLiLuMoVe"):
    rows = all_site_preds(name)
    for rule in ("E1", "E2", "E3"):
        errs = [abs(choose(c, rule) - y) for y, c in rows if c]
        print(f"  {name:11s} {rule}: MAE {np.mean(errs):.3f}  "
              f"(scored {len(errs)}/{len(rows)}; molecules with no SMARTS site excluded)")


# E4: learned site ranker. A classifier scores each candidate site
# (site feats + predicted pKa + priority rank) for "is this the site
# whose pKa was measured"; highest score wins. Trained on the training
# set's annotations with 5-fold out-of-fold predicted pKas so the
# ranker never sees in-sample regressor output.
from sklearn.model_selection import KFold

train_mols = [m for m in Chem.SDMolSupplier(f"{D}/combined_training_datasets_unique.sdf") if m is not None]
oof = np.zeros(len(yc))
for tr, va in KFold(5, shuffle=True, random_state=0).split(Xc):
    oof_m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                              colsample_bytree=0.5, verbose=-1, random_state=0).fit(Xc[tr], yc[tr])
    oof[va] = oof_m.predict(Xc[va])


def rank_feats(n, s, rank, pred):
    kind = "acidic" if s.kind == "acid" else "basic"
    return np.concatenate([site_feats(n, s.atom, kind), [pred, rank, s.kind == "acid"]])


RX, RY = [], []
for i, m in enumerate(train_mols):
    p = m.GetPropsAsDict()
    n = neutralize(m)
    sites = find_sites(n)
    if len(sites) < 2 or p["marvin_atom"] not in [s.atom for s in sites]:
        continue
    g = glob(n)
    for rank, s in enumerate(sites):
        kind = "acidic" if s.kind == "acid" else "basic"
        pred = float(mc.predict(np.concatenate([g, site_feats(n, s.atom, kind)]).reshape(1, -1))[0])
        # use OOF value for the annotated site to avoid in-sample optimism
        if s.atom == p["marvin_atom"]:
            pred = oof[i]
        RX.append(rank_feats(n, s, rank, pred))
        RY.append(s.atom == p["marvin_atom"])
ranker = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                            verbose=-1, random_state=0).fit(np.array(RX), np.array(RY))
print(f"  ranker trained on {sum(RY)} multi-site molecules")
for name in ("novartis", "AvLiLuMoVe"):
    rows = all_site_preds(name)
    errs = []
    for (y, cands), m in zip(rows, [m for m in Chem.SDMolSupplier(
            f"{D}/{name}_cleaned_mono_unique_notraindata.sdf") if m is not None]):
        if not cands:
            continue
        if len(cands) == 1:
            errs.append(abs(cands[0][1] - y))
            continue
        n = neutralize(m)
        sc = ranker.predict_proba(np.array([rank_feats(n, s, r, v)
                                            for r, (s, v) in enumerate(cands)]))[:, 1]
        errs.append(abs(cands[int(np.argmax(sc))][1] - y))
    print(f"  {name:11s} E4 (learned ranker): MAE {np.mean(errs):.3f}  (scored {len(errs)}/{len(rows)})")
