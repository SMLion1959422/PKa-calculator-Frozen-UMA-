"""
benchmark_predict_smart_v3.py

Two-tier benchmark, informed by timing pilot: site disambiguation is
near-free (~0-9s/molecule) so runs on the full set; tautomer
correction is expensive and bimodal (0.01s-230s/molecule, ~45s avg)
so runs on a stratified subsample instead of all 403.
"""
import random
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
SUBSAMPLE_N = 30  # per dataset, so 60 total across novartis + avlilumove
SEED = 42


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


def score(predictor, smi, exp, use_tautomer, use_site):
    try:
        plain = predictor.predict(smi)
    except Exception:
        plain = None
    try:
        r = predictor.predict_smart(smi, use_tautomer_correction=use_tautomer,
                                     use_site_disambiguation=use_site)
        smart = r["pKa"]
        taut = r["tautomer_applied"]
        site = r["site_disambiguation_applied"]
    except Exception:
        smart, taut, site = None, False, False
    return {
        "smiles": smi, "exp": exp,
        "plain_pred": plain, "smart_pred": smart,
        "plain_err": abs(plain - exp) if plain is not None else None,
        "smart_err": abs(smart - exp) if smart is not None else None,
        "tautomer_applied": taut, "site_disambiguation_applied": site,
    }


def report(df, label):
    both = df.dropna(subset=["plain_err", "smart_err"])
    print(f"\n=== {label} (n={len(df)}, both-succeeded n={len(both)}) ===")
    if len(both) == 0:
        print("  no successful pairs")
        return
    print(f"  predict()       MAE: {both['plain_err'].mean():.3f}")
    print(f"  predict_smart() MAE: {both['smart_err'].mean():.3f}")
    print(f"  tautomer applied: {int(df['tautomer_applied'].sum())}   "
          f"site applied: {int(df['site_disambiguation_applied'].sum())}")
    changed = both[(both["plain_pred"] - both["smart_pred"]).abs() > 0.05].copy()
    if len(changed):
        changed["improved"] = changed["smart_err"] < changed["plain_err"]
        print(f"  changed: {len(changed)} ({int(changed['improved'].sum())} improved, "
              f"{int((~changed['improved']).sum())} worse)")


def main():
    print("loading UMA + model_core_v3...")
    predictor = PkaPredictor(MODEL_PATH)
    print("loaded.\n")

    sets = pd.concat([load_set(p, n) for n, p in SDF_PATHS.items()]).reset_index(drop=True)
    print(f"loaded {len(sets)} molecules total\n")

    # --- Tier 1: full set, site disambiguation only (fast) ---
    print("=== TIER 1: full set, site-disambiguation only ===")
    rows = [score(predictor, r.smiles, r.exp, use_tautomer=False, use_site=True)
            for r in tqdm(sets.itertuples(), total=len(sets))]
    site_only = pd.DataFrame(rows)
    site_only.to_csv("benchmark_site_only_full.csv", index=False)
    report(site_only, "TIER 1: site-only, full 403")

    # --- Tier 2: stratified subsample, both corrections (slow) ---
    random.seed(SEED)
    sub = (sets.groupby("dataset", group_keys=False)
                .apply(lambda g: g.sample(min(SUBSAMPLE_N, len(g)), random_state=SEED)))
    print(f"\n=== TIER 2: {len(sub)}-molecule subsample, tautomer + site both on ===")
    rows2 = [score(predictor, r.smiles, r.exp, use_tautomer=True, use_site=True)
             for r in tqdm(sub.itertuples(), total=len(sub))]
    both_on = pd.DataFrame(rows2)
    both_on.to_csv("benchmark_both_subsample.csv", index=False)
    report(both_on, "TIER 2: tautomer+site, subsample")


if __name__ == "__main__":
    main()
