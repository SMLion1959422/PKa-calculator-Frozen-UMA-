"""Fetches additional experimental pKa data from czodrowskilab's newer
Multiprotic-pKa-Processing repo (MIT licensed, same research group whose
chembl25+datawarrior data your project already trains on:
https://github.com/czodrowskilab/Multiprotic-pKa-Processing).

We do NOT run their full processing pipeline (it needs commercial
ChemAxon Marvin + OpenEye QUACPAC for protonation-SITE annotation) -
your project already does its own site detection via SMARTS, so we only
need the raw (SMILES, pKa) pairs from their datasets/*.sdf files, read
the same way your existing train_core_v3.py already reads
combined_training_datasets_unique.sdf.

Datasets pulled (datawarrior.sdf skipped - near-certainly already
covered by what you have):
  - chembl26.sdf          (newer ChEMBL version than your chembl25)
  - hunt_et_al.sdf         (Hunt et al. 2020, semi-empirical QM + RBF)
  - literature_compilation.sdf  (compiled from multiple publications)
  - settimo_et_al.sdf      (Settimo, Bellman & Knegtel 2014)
  - sampl6.sdf              (SAMPL6 challenge - NOTE: has its OWN
                              separate license file, distinct from the
                              repo's blanket MIT license - check
                              datasets/sampl6.LICENSE yourself before
                              using this one if you have any licensing
                              concerns; the others are plain MIT)

CRITICAL: explicitly excludes any molecule whose canonical SMILES
matches something in your Novartis or AvLiLuMoVe TEST sets, so this
can't quietly leak test molecules into training. Also excludes anything
already in your existing training set, so we only add genuinely NEW
molecules and don't waste embedding compute re-adding what you have.
"""
import subprocess
import os
import gzip
import shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

REPO_URL = "https://github.com/czodrowskilab/Multiprotic-pKa-Processing.git"
CLONE_DIR = "extra_pka_source"
NEW_DATASETS = [
    "chembl26.sdf.gz",
    "hunt_et_al.sdf.gz",
    "literature_compilation.sdf.gz",
    "settimo_et_al.sdf.gz",
    "sampl6.sdf.gz",   # see license note above
]

def decompress_if_needed(gz_path):
    """.sdf.gz -> .sdf (once), so PandasTools.LoadSDF can read it -
    LoadSDF doesn't handle gzip directly."""
    sdf_path = gz_path[:-3]   # strip ".gz"
    if os.path.exists(sdf_path):
        return sdf_path
    if not os.path.exists(gz_path):
        return None
    with gzip.open(gz_path, "rb") as f_in, open(sdf_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return sdf_path

def clone_if_needed():
    if os.path.isdir(CLONE_DIR):
        print(f"{CLONE_DIR}/ already exists, skipping clone "
              f"(delete it to re-fetch)")
        return
    print(f"cloning {REPO_URL} (shallow, no submodules needed)...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
                    check=True)

def load_sdf_pka_table(path, source_name):
    if not os.path.exists(path):
        print(f"  SKIP {source_name}: file not found at {path}")
        return pd.DataFrame(columns=["smiles", "pKa", "source"])
    df = PandasTools.LoadSDF(path)
    pk_col = next((c for c in df.columns
                    if c.lower() in ("pka", "pka_value", "value", "pka1")), None)
    if pk_col is None:
        print(f"  SKIP {source_name}: no recognizable pKa column "
              f"(columns were: {list(df.columns)})")
        return pd.DataFrame(columns=["smiles", "pKa", "source"])
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
        rows.append({"smiles": smi, "pKa": v, "source": source_name})
    out = pd.DataFrame(rows).drop_duplicates("smiles")
    print(f"  {source_name}: {len(out)} valid (SMILES, pKa) pairs")
    return out

def load_smiles_set_from_sdf(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found - cannot guard against this "
              f"as a leakage source, please verify manually")
        return set()
    df = PandasTools.LoadSDF(path)
    out = set()
    for _, r in df.iterrows():
        m = r.get("ROMol")
        if m is None:
            continue
        try:
            out.add(Chem.MolToSmiles(m))
        except Exception:
            pass
    return out

clone_if_needed()

print("\nloading new candidate datasets...")
tables = []
for fn in NEW_DATASETS:
    gz_path = f"{CLONE_DIR}/datasets/{fn}"
    sdf_path = decompress_if_needed(gz_path)
    if sdf_path is None:
        print(f"  SKIP {fn}: not found at {gz_path}")
        tables.append(pd.DataFrame(columns=["smiles", "pKa", "source"]))
        continue
    tables.append(load_sdf_pka_table(sdf_path, fn))
candidates = pd.concat(tables, ignore_index=True).drop_duplicates("smiles")
print(f"\ntotal unique candidate molecules across all new sources: {len(candidates)}")

print("\nloading existing training set (to avoid redundant re-adding)...")
existing_train = load_smiles_set_from_sdf(
    "mlpka/datasets/combined_training_datasets_unique.sdf")
print(f"  {len(existing_train)} molecules already in your training set")

print("\nloading held-out TEST sets (critical leakage guard)...")
novartis_smiles = load_smiles_set_from_sdf(
    "mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf")
avlilumove_smiles = load_smiles_set_from_sdf(
    "mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf")
test_smiles = novartis_smiles | avlilumove_smiles
print(f"  {len(test_smiles)} molecules across both test sets")

leaked = candidates[candidates.smiles.isin(test_smiles)]
if len(leaked) > 0:
    print(f"\n*** EXCLUDING {len(leaked)} candidate molecules that overlap "
          f"with your Novartis/AvLiLuMoVe TEST sets - these would have "
          f"silently corrupted your evaluation if added to training ***")

net_new = candidates[
    ~candidates.smiles.isin(test_smiles) &
    ~candidates.smiles.isin(existing_train)
].reset_index(drop=True)

print(f"\n=== RESULT ===")
print(f"candidates found:        {len(candidates)}")
print(f"already in training:     {len(candidates) - len(net_new) - len(leaked)}")
print(f"excluded (test overlap): {len(leaked)}")
print(f"genuinely NEW molecules: {len(net_new)}")
print(f"\nby source (of the net-new ones):")
print(net_new.source.value_counts())

net_new[["smiles", "pKa", "source"]].to_csv("extra_pka_data.csv", index=False)
print(f"\nsaved -> extra_pka_data.csv ({len(net_new)} rows)")
print("next: run embed_extra_data.py to embed only these new molecules "
      "(reuses your existing feat_train_v3.pkl cache for everything else)")
