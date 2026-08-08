"""Locate the AttributeError by testing the model-loading and predict
path directly, with the traceback exposed."""
import sys, traceback, joblib
import numpy as np
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

for path in ["models/model_core_v16_elec.pkl",
             "models/model_core_v18_maxdata.pkl",
             "models/model_core_v11.pkl"]:
    try:
        b = joblib.load(path)
        print(f"{path}")
        print(f"   keys: {list(b.keys())}")
        for k, v in b.items():
            print(f"     {k:14s} {type(v).__name__}")
    except Exception as e:
        print(f"{path}  -> {type(e).__name__}: {e}")
    print()

print("=== live predict test on v16 ===")
b = joblib.load("models/model_core_v16_elec.pkl")
try:
    gbm, ridge, sc, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]
    print(f"  unpacked ok, blend_w={bw}")
    n = sc.mean_.shape[0]
    print(f"  scaler expects {n} features")
    feat = np.zeros((1, n))
    g = gbm.predict(feat)
    print(f"  gbm.predict ok -> {g}")
    r = ridge.predict(sc.transform(feat))
    print(f"  ridge.predict ok -> {r}")
    raw = (1-bw)*g[0] + bw*r[0]
    print(f"  blended raw = {raw}")
    out = cal.predict([raw])
    print(f"  calibrator.predict ok -> {out}")
    print("\n  ENTIRE PREDICT PATH WORKS")
except Exception:
    traceback.print_exc()
