"""Breaks the already-saved characterization_external_v3.csv down by
dataset (novartis vs avlilumove) - eval_core_v3.py computed this but
only printed the pooled overall number. No re-run needed, this just
reads the CSV eval_core_v3.py already wrote.
"""
import pandas as pd

d = pd.read_csv("characterization_external_v3.csv")

print("=== v3 MAE BY DATASET (like-for-like vs RESULTS.md/README) ===")
print(d.groupby("dataset")["err"].agg(["mean", "count"]).round(3))
print()
print("RESULTS.md reported (v2, global-only, clean retrain):")
print("  novartis   (n=263): overall MAE 1.16")
print("  avlilumove         : no single overall MAE stated in RESULTS.md -")
print("                       only broken down by size there (0.46/0.60/0.64/1.05")
print("                       for <15/15-22/22-30/>30 heavy atoms)")
