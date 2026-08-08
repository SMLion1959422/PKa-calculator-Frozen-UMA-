"""
benchmark_predict_smart.py

Compares predict() vs predict_smart() MAE on external test sets, to
check whether the tautomer/site-disambiguation corrections actually
help on average (not just on the hand-picked validation cases).
"""
import numpy as np
import pandas as pd
from umapka.predictor import PkaPredictor

# ---- CONFIG: adjust these to match your repo ----
MODEL_PATH = "model_core_v3.pkl"   # swap to your v4 file if you saved one separately

DATASETS = {
    # name: (csv_path, smiles_column, true_pka_column)
    "Novartis":    ("data/novartis_test.csv",    "smiles", "pKa"),
    "AvLiLuMoVe":  ("data/avlilumove_test.csv",  "smiles", "pKa"),
}
# ---------------------------------------------------


def load_dataset(path, smiles_col, pka_col):
    df = pd.read_csv(path)
    if smiles_col not in df.columns or pka_col not in df.columns:
        raise ValueError(
            f"Couldn't find columns {smiles_col!r}/{pka_col!r} in {path}. "
            f"Actual columns: {list(df.columns)}. "
            f"Edit the DATASETS dict at the top of this script."
        )
    return df[[smiles_col, pka_col]].rename(
        columns={smiles_col: "smiles", pka_col: "true_pka"})


def run_benchmark(predictor, df, name):
    rows = []
    n_fail_plain = n_fail_smart = 0
    n_tautomer = n_site = 0

    for i, row in df.iterrows():
        smi, true_pka = row["smiles"], row["true_pka"]

        try:
            plain = predictor.predict(smi)
        except Exception:
            n_fail_plain += 1
            plain = None

        try:
            smart_result = predictor.predict_smart(smi)
            smart = smart_result["pKa"]
            if smart_result["tautomer_applied"]:
                n_tautomer += 1
            if smart_result["site_disambiguation_applied"]:
                n_site += 1
        except Exception:
            n_fail_smart += 1
            smart = None

        rows.append({
            "smiles": smi,
            "true_pka": true_pka,
            "plain_pred": plain,
            "smart_pred": smart,
            "plain_err": abs(plain - true_pka) if plain is not None else None,
            "smart_err": abs(smart - true_pka) if smart is not None else None,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{name}] {i + 1}/{len(df)} done...")

    out = pd.DataFrame(rows)
    both = out.dropna(subset=["plain_err", "smart_err"])
    mae_plain = both["plain_err"].mean()
    mae_smart = both["smart_err"].mean()

    print(f"\n=== {name} (n={len(df)}, both-succeeded n={len(both)}) ===")
    print(f"  predict()       MAE: {mae_plain:.4f}")
    print(f"  predict_smart() MAE: {mae_smart:.4f}")
    print(f"  delta (smart - plain): {mae_smart - mae_plain:+.4f} "
          f"({'better' if mae_smart < mae_plain else 'worse or equal'})")
    print(f"  tautomer correction applied: {n_tautomer}/{len(df)}")
    print(f"  site disambiguation applied: {n_site}/{len(df)}")
    print(f"  predict() failures: {n_fail_plain}   predict_smart() failures: {n_fail_smart}")

    changed = both[(both["plain_pred"] - both["smart_pred"]).abs() > 0.05].copy()
    if len(changed):
        changed["improved"] = changed["smart_err"] < changed["plain_err"]
        print(f"  molecules where prediction changed: {len(changed)} "
              f"({int(changed['improved'].sum())} improved, "
              f"{int((~changed['improved']).sum())} got worse)")

    out_path = f"benchmark_{name.lower()}_results.csv"
    out.to_csv(out_path, index=False)
    print(f"  full per-molecule results -> {out_path}")

    return mae_plain, mae_smart


def main():
    print("loading UMA + model...")
    predictor = PkaPredictor(MODEL_PATH)
    print("loaded.\n")

    summary = []
    for name, (path, smiles_col, pka_col) in DATASETS.items():
        df = load_dataset(path, smiles_col, pka_col)
        mae_plain, mae_smart = run_benchmark(predictor, df, name)
        summary.append((name, mae_plain, mae_smart))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, mp, ms in summary:
        winner = "predict_smart()" if ms < mp else "predict()"
        print(f"  {name:15s}  plain={mp:.4f}  smart={ms:.4f}  delta={ms - mp:+.4f}  ({winner} wins)")


if __name__ == "__main__":
    main()
