"""
benchmark_predict_smart_v2.py

Compares predict() vs predict_smart() MAE on the same Novartis/
AvLiLuMoVe SDF files and pKa filtering used by eval_core_v3.py, using
model_core_v3.pkl - the only model currently wired into the standard
features()/predict() interface that predict_smart() relies on.
(model_core_v4.pkl uses a separate state_features_v4() pipeline not
compatible with predict()/predict_smart() - see eval_core_v4.py.)
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm
from umapka import PkaPredictor

MODEL_PATH = "models/model_core_v3.pkl"

SDF_PATHS = {
    "novartis": "mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf",
    "avlilumove": "mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf",
}


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


def main():
    print("loading UMA + model_core_v3...")
    predictor = PkaPredictor(MODEL_PATH)
    print("loaded.\n")

    sets = pd.concat([load_set(p, n) for n, p in SDF_PATHS.items()]).reset_index(drop=True)
    print(f"loaded {len(sets)} molecules total across {list(SDF_PATHS)}\n")

    rows = []
    n_fail_plain = n_fail_smart = 0
    n_tautomer = n_site = 0

    for r in tqdm(sets.itertuples(), total=len(sets)):
        try:
            plain = predictor.predict(r.smiles)
        except Exception:
            n_fail_plain += 1
            plain = None

        try:
            smart_result = predictor.predict_smart(r.smiles)
            smart = smart_result["pKa"]
            if smart_result["tautomer_applied"]:
                n_tautomer += 1
            if smart_result["site_disambiguation_applied"]:
                n_site += 1
        except Exception:
            n_fail_smart += 1
            smart = None

        rows.append({
            "dataset": r.dataset,
            "smiles": r.smiles,
            "exp": r.exp,
            "plain_pred": plain,
            "smart_pred": smart,
            "plain_err": abs(plain - r.exp) if plain is not None else None,
            "smart_err": abs(smart - r.exp) if smart is not None else None,
        })

    out = pd.DataFrame(rows)
    out.to_csv("benchmark_predict_smart_v3_results.csv", index=False)

    both = out.dropna(subset=["plain_err", "smart_err"])

    print(f"\n=== OVERALL (n={len(out)}, both-succeeded n={len(both)}) ===")
    print(f"  predict()       MAE: {both['plain_err'].mean():.3f}")
    print(f"  predict_smart() MAE: {both['smart_err'].mean():.3f}")
    print(f"  (v3 baseline per RESULTS.md: novartis=1.018, avlilumove=0.421)\n")

    print("=== BY DATASET ===")
    print(both.groupby("dataset")[["plain_err", "smart_err"]].agg(["mean", "count"]).round(3))

    print(f"\n  tautomer correction applied: {n_tautomer}/{len(out)}")
    print(f"  site disambiguation applied: {n_site}/{len(out)}")
    print(f"  predict() failures: {n_fail_plain}   predict_smart() failures: {n_fail_smart}")

    changed = both[(both["plain_pred"] - both["smart_pred"]).abs() > 0.05].copy()
    if len(changed):
        changed["improved"] = changed["smart_err"] < changed["plain_err"]
        print(f"\n  molecules where prediction changed: {len(changed)} "
              f"({int(changed['improved'].sum())} improved, "
              f"{int((~changed['improved']).sum())} got worse)")

    print("\nfull per-molecule results -> benchmark_predict_smart_v3_results.csv")


if __name__ == "__main__":
    main()
