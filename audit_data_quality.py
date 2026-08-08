"""Data-quality audit - no GPU, no UMA, just RDKit/pandas. Three checks
nobody has run yet in this whole project, any of which could matter
more than another pooling/architecture tweak:

1. CONFLICTING DUPLICATE LABELS: does the same molecule appear more
   than once across your merged training sources with meaningfully
   different reported pKa values? This is pure label noise - if a
   molecule is labeled 4.1 in one source and 6.8 in another, no amount
   of model capacity fixes that, and it's actively teaching the model
   contradictory things.

2. pKa RANGE COVERAGE: histogram of training-label pKa values. Directly
   checks the "sparse outside 2-12" limitation mentioned in your paper
   - now with actual numbers instead of an assumption.

3. TRAIN/TEST SIZE-DISTRIBUTION MISMATCH: is your training set
   systematically smaller-molecule-weighted than Novartis/AvLiLuMoVe?
   If so, that alone could explain part of the external-MAE gap,
   independent of any pooling architecture - the model would have
   genuinely seen fewer large molecules to learn from, which is a data
   problem, not an architecture problem.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

def load_sdf(path, name):
    df = PandasTools.LoadSDF(path)
    pk_col = next((c for c in df.columns
                    if c.lower() in ("pka", "pka_value", "value", "pka1")), None)
    rows = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None:
            continue
        try:
            v = float(r[pk_col])
        except Exception:
            continue
        if not (0 < v < 14):
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        rows.append({"smiles": smi, "pKa": v, "source": name,
                      "n_atoms": m.GetNumAtoms()})
    return pd.DataFrame(rows)

print("=" * 60)
print("CHECK 1: conflicting duplicate labels")
print("=" * 60)

train = load_sdf("mlpka/datasets/combined_training_datasets_unique.sdf", "combined")
try:
    extra = pd.read_csv("extra_pka_data.csv")
    extra["n_atoms"] = extra.smiles.apply(
        lambda s: Chem.MolFromSmiles(s).GetNumAtoms() if Chem.MolFromSmiles(s) else None)
    all_data = pd.concat([train, extra], ignore_index=True)
    print("(including extra_pka_data.csv in this check)")
except FileNotFoundError:
    all_data = train
    print("(extra_pka_data.csv not found - checking combined_training_datasets_unique.sdf only)")

dupe_groups = all_data.groupby("smiles").filter(lambda g: len(g) > 1).groupby("smiles")
conflicts = []
for smi, g in dupe_groups:
    spread = g["pKa"].max() - g["pKa"].min()
    if spread > 0.5:   # meaningfully different, not just rounding noise
        conflicts.append({"smiles": smi, "n_labels": len(g),
                           "pKa_values": sorted(g["pKa"].tolist()),
                           "spread": spread,
                           "sources": sorted(g["source"].unique().tolist())})

conflicts_df = pd.DataFrame(conflicts)
print(f"\n{len(dupe_groups.groups)} molecules appear more than once in your data")
if len(conflicts_df) > 0:
    conflicts_df = conflicts_df.sort_values("spread", ascending=False)
print(f"{len(conflicts_df)} of those have CONFLICTING labels (>0.5 pKa units apart)")
if len(conflicts_df) > 0:
    print(f"\nworst 10 conflicts:")
    print(conflicts_df.head(10).to_string(index=False))
    conflicts_df.to_csv("label_conflicts.csv", index=False)
    print(f"\nfull list saved -> label_conflicts.csv "
          f"({len(conflicts_df)} rows, ~{len(conflicts_df)/len(all_data)*100:.1f}% "
          f"of your unique molecule count)")
    print("\nRECOMMENDATION: for each conflict, either pick the more reliable")
    print("source, average the values, or drop the molecule entirely. Even")
    print(f"{len(conflicts_df)} noisy labels can measurably hurt a model trained")
    print("on only ~5,500 total examples.")
else:
    print("\nNo significant conflicts found - your merged data is clean on this axis.")

print("\n" + "=" * 60)
print("CHECK 2: pKa range coverage")
print("=" * 60)
bins = [0, 2, 4, 6, 8, 10, 12, 14]
labels_ = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14"]
train_unique = all_data.drop_duplicates("smiles")
train_unique["bin"] = pd.cut(train_unique["pKa"], bins=bins, labels=labels_)
print(train_unique["bin"].value_counts().sort_index())
print(f"\ntotal unique training molecules: {len(train_unique)}")
print(f"outside the 2-12 'core' range: "
      f"{(~train_unique.pKa.between(2,12)).sum()} "
      f"({(~train_unique.pKa.between(2,12)).mean()*100:.1f}%)")

print("\n" + "=" * 60)
print("CHECK 3: train vs. test molecule-size distribution")
print("=" * 60)
def load_smiles_sizes(path):
    df = PandasTools.LoadSDF(path)
    sizes = []
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is not None:
            sizes.append(m.GetNumAtoms())
    return np.array(sizes)

novartis_sizes = load_smiles_sizes("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf")
avli_sizes = load_smiles_sizes("mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf")
train_sizes = train_unique["n_atoms"].values

for name, sizes in [("training set", train_sizes),
                     ("Novartis (test)", novartis_sizes),
                     ("AvLiLuMoVe (test)", avli_sizes)]:
    print(f"{name:20s}: median={np.median(sizes):.0f}  "
          f"mean={np.mean(sizes):.1f}  "
          f">30 atoms: {(sizes>30).mean()*100:.1f}%  "
          f"(n={len(sizes)})")

train_frac_large = (train_sizes > 30).mean()
test_frac_large = (np.concatenate([novartis_sizes, avli_sizes]) > 30).mean()
print(f"\ntraining set is {train_frac_large*100:.1f}% large molecules (>30 atoms)")
print(f"test sets are {test_frac_large*100:.1f}% large molecules")
if test_frac_large > train_frac_large * 1.3:
    print("\n*** Test sets are meaningfully MORE weighted toward large ***")
    print("*** molecules than training. Part of the large-molecule error ***")
    print("*** could be a genuine data-representation gap, not (only) an ***")
    print("*** architecture problem. ***")
else:
    print("\nSize distributions are reasonably comparable - this specific")
    print("mismatch is probably not a major contributor to the gap.")
