"""
time_predict_smart.py - isolates which step of predict_smart() is
slow, on a small sample, before committing to the full 403-molecule run.
"""
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from umapka import PkaPredictor

MODEL_PATH = "models/model_core_v3.pkl"
N_SAMPLE = 15

def load_set(path, name):
    df = PandasTools.LoadSDF(path)
    pk_col = next(c for c in df.columns if c.lower() in ("pka", "pka_value", "value"))
    out = []
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
            out.append({"dataset": name, "smiles": Chem.MolToSmiles(m), "exp": v})
        except Exception:
            pass
    return pd.DataFrame(out).drop_duplicates("smiles")

print("loading UMA + model_core_v3...")
predictor = PkaPredictor(MODEL_PATH)
print("loaded.\n")

sets = load_set("mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf", "novartis")
sample = sets.head(N_SAMPLE)

for r in sample.itertuples():
    t0 = time.time()
    try:
        plain = predictor.predict(r.smiles)
    except Exception as e:
        plain = f"FAIL: {e}"
    t1 = time.time()

    try:
        tinfo = predictor.rank_tautomers_safe(r.smiles)
    except Exception as e:
        tinfo = f"FAIL: {e}"
    t2 = time.time()

    try:
        sinfo = predictor.rank_same_type_sites(r.smiles)
    except Exception as e:
        sinfo = f"FAIL: {e}"
    t3 = time.time()

    try:
        smart = predictor.predict_smart(r.smiles)["pKa"]
    except Exception as e:
        smart = f"FAIL: {e}"
    t4 = time.time()

    print(f"{r.smiles[:40]:42s} predict={t1-t0:5.2f}s  tautomer={t2-t1:5.2f}s  "
          f"site={t3-t2:5.2f}s  smart_total={t4-t3:5.2f}s")

print(f"\nprojected full-set time at these rates: "
      f"predict_smart ~{(sample.shape[0] and 0) or ''}")
