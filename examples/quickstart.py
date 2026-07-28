"""Minimal usage example for umapka."""
from umapka import PkaPredictor

p = PkaPredictor("models/model_core.pkl")

print("--- monoprotic (validated range) ---")
for name, smi, exp in [
    ("acetic acid",  "CC(=O)O",                          4.76),
    ("benzoic acid", "OC(=O)c1ccccc1",                   4.20),
    ("phenol",       "Oc1ccccc1",                        9.95),
    ("ethylamine",   "CCN",                             10.70),
    ("pyridine",     "c1ccncc1",                         5.23),
    ("ibuprofen",    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",       4.90),
]:
    pred = p.predict(smi)
    print(f"{name:<15} pred {pred:5.2f}   exp {exp:5.2f}   err {abs(pred-exp):.2f}")

print("\n--- choosing a site explicitly ---")
smi = "CC(=O)Nc1ccc(O)cc1"          # paracetamol
for s in p.sites(smi):
    print(f"  [{s['index']}] {s['group']:<16} atom {s['atom']:<3} "
          f"-> pKa {p.predict_site(smi, s['index']):.2f}")
print("  (experimental phenol pKa: 9.38)")
