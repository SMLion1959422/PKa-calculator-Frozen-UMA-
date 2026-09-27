"""Fit and validate the site-site coupling constant D in
umapka.microstates on symmetric diacids / diamines.

For a molecule with two identical sites the intrinsic pKa cancels:
    pKa2 - pKa1 = log10(4) + W
so W (and hence D) is tested with no dependence on any ML model.
We fit D by 1-D least squares on the macro-pKa gap computed by the
real MicrostateModel, report leave-one-out error, then predict
amino acids (not used in the fit) from their experimental
microconstants as a held-out check.

Data: data/symmetric_polyprotic.csv - 25 C, low ionic strength,
values from standard compilations (CRC Handbook; Perrin, Dissociation
Constants of Organic Bases). Several entries differ between
compilations by 0.1-0.3 units (noted in the file); treat the MAE
floor accordingly.

Run from the repo root:  python dev/validate_coupling.py
"""
import csv
import numpy as np
from scipy.optimize import minimize_scalar
from umapka import microstates as ms
from umapka.sites import find_sites, neutralize
from rdkit import Chem

rows = list(csv.DictReader(open("data/symmetric_polyprotic.csv")))
geoms = []
for r in rows:
    mol = neutralize(Chem.MolFromSmiles(r["smiles"]))
    sites = find_sites(mol)
    assert len(sites) == 2, r["name"]
    geoms.append((mol, sites, ms.site_geometry(mol, sites)))
gap_exp = np.array([float(r["pKa2"]) - float(r["pKa1"]) for r in rows])


def gaps(D, idx):
    out = []
    for k in idx:
        mol, sites, g = geoms[k]
        W = ms.coupling_matrix(g, 2, D=D)
        m = ms.MicrostateModel(r["smiles"], sites, np.array([5.0, 5.0]), W, _mol=mol)
        p = m.macro_pkas()
        out.append(p[1] - p[0])
    return np.array(out)


def fit(idx):
    f = lambda D: np.sum((gaps(D, idx) - gap_exp[idx]) ** 2)
    return minimize_scalar(f, bounds=(2, 60), method="bounded").x


all_idx = np.arange(len(rows))
D = fit(all_idx)
pred = gaps(D, all_idx)
loo = np.array([gaps(fit(np.delete(all_idx, i)), [i])[0] for i in all_idx])
base = np.array([np.delete(gap_exp, i).mean() for i in all_idx])
no_coupling = np.log10(4)

print(f"fitted D = {D:.2f}   (eps(r) = D*r, capped at bulk water)")
print(f"{'molecule':22s}{'exp gap':>9}{'fit':>7}{'LOO':>7}")
for r, e, p, l in zip(rows, gap_exp, pred, loo):
    print(f"{r['name']:22s}{e:9.2f}{p:7.2f}{l:7.2f}")
print(f"\nMAE of pKa2-pKa1 gap (n={len(rows)}):")
print(f"  independent sites (statistical log4 only) {np.abs(no_coupling - gap_exp).mean():.2f}")
print(f"  constant gap (LOO mean)                   {np.abs(base - gap_exp).mean():.2f}")
print(f"  microstate model, in-sample               {np.abs(pred - gap_exp).mean():.2f}")
print(f"  microstate model, leave-one-out           {np.abs(loo - gap_exp).mean():.2f}")

# held-out: amino acids from experimental MICRO constants. The intrinsic
# (other-site-neutral) micro pKas below are derived from the
# microconstant cycle; the model must reproduce the MACRO pKas.
# glycine: macro 2.35/9.78 and zwitterion/neutral tautomer ratio
# log Kz ~ 5.4 give pk(COOH | NH2) = 4.38 and pk(NH3+ | COOH) = 7.75.
print("\nheld-out zwitterion check (glycine, exp macro pKa 2.35 / 9.78):")
g = ms.build_model("NCC(=O)O", {0: 4.38, 1: 7.75}, D=D)
print(f"  independent-site answer: 4.38 / 7.75   model: "
      f"{g.macro_pkas()[0]:.2f} / {g.macro_pkas()[1]:.2f}   pI {g.isoelectric_point():.2f} (exp 6.06)")
print(f"  dominant species at pH 7: {g.species(7.0, 0.05)[0]}")
