"""1) EXPANDED SITE COVERAGE - roughly doubles the ionizable groups the
tool recognizes. Every pattern is validated before being added; invalid
SMARTS are reported and skipped rather than silently breaking matching.

Appends to ACID_SITES/BASE_SITES in umapka/predictor.py."""
import re, sys
sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

NEW_ACIDS = [
    ("boronic_acid",      "[BX3]([OX2H1])[OX2H1]", 1),
    ("phosphonic_acid",   "[PX4](=O)([OX2H1])[OX2H1]", 2),
    ("phosphinic_acid",   "[PX4](=O)([OX2H1])[#6]", 2),
    ("sulfinic_acid",     "[SX3](=O)[OX2H1]", 2),
    ("acyl_sulfonamide",  "[CX3](=O)[NX3H1][SX4](=O)(=O)", 2),
    ("sulfonimide",       "[SX4](=O)(=O)[NX3H1][SX4](=O)(=O)", 3),
    ("hydantoin",         "[NX3H1]([CX3]=O)[CX3](=O)[NX3]", 0),
    ("barbiturate_NH",    "O=[CX3][NX3H1][CX3]=O", 2),
    ("thiourea_NH",       "[NX3H1][CX3](=[SX1])[NX3]", 0),
    ("carbamate_NH",      "[NX3H1][CX3](=O)[OX2]", 0),
    ("enol",              "[CX3]=[CX3][OX2H1]", 2),
    ("thiophenol",        "[c][SX2H1]", 1),
    ("hydroxylamine_OH",  "[NX3][OX2H1]", 1),
    ("pyrazole_NH",       "[nX3H1]1[nX2][cX3][cX3][cX3]1", 0),
    ("imidazole_NH",      "[nX3H1]1[cX3][nX2][cX3][cX3]1", 0),
    ("triazole_NH",       "[nX3H1]1[nX2][nX2][cX3][cX3]1", 0),
    ("benzimidazole_NH",  "[nX3H1]1[cX3][nX2]c2ccccc21", 0),
    ("purine_NH",         "[nX3H1]1[cX3][nX2][cX3]2[cX3]1[nX2][cX3][nX2][cX3]2", 0),
    ("squaric_acid",      "[OX2H1][CX3]1=[CX3][CX3](=O)[CX3]1=O", 0),
    ("vinylogous_acid",   "[OX2H1][CX3]=[CX3][CX3]=O", 0),
    ("alpha_nitro_CH",    "[CX4;H1,H2][NX3](=O)=O", 0),
    ("sulfone_CH",        "[CX4;H1,H2][SX4](=O)(=O)", 0),
    ("nitramide",         "[NX3H1][NX3](=O)=O", 0),
]

NEW_BASES = [
    ("imidazole_N",       "[nX2]1[cX3][nX3H1][cX3][cX3]1", 0),
    ("pyrimidine_N",      "[nX2]1[cX3][nX2][cX3][cX3][cX3]1", 0),
    ("pyrazine_N",        "[nX2]1[cX3][cX3][nX2][cX3][cX3]1", 0),
    ("oxazole_N",         "[nX2]1[cX3][oX2][cX3][cX3]1", 0),
    ("thiazole_N",        "[nX2]1[cX3][sX2][cX3][cX3]1", 0),
    ("triazine_N",        "[nX2]1[cX3][nX2][cX3][nX2][cX3]1", 0),
    ("imine_N",           "[NX2;H0,H1]=[CX3]", 0),
    ("hydrazine_N",       "[NX3;H1,H2][NX3;H1,H2]", 0),
    ("hydroxylamine_N",   "[NX3;H1,H2][OX2H1]", 0),
    ("phosphazene_N",     "[NX2]=[PX4]", 0),
    ("piperazine_N",      "[NX3;H1]1[CX4][CX4][NX3][CX4][CX4]1", 0),
    ("morpholine_N",      "[NX3;H0,H1]1[CX4][CX4][OX2][CX4][CX4]1", 0),
]

path = "umapka/predictor.py"
src = open(path, encoding="utf-8").read()

def render(entries, label):
    ok, bad = [], []
    for name, sm, ai in entries:
        if Chem.MolFromSmarts(sm) is None:
            bad.append((name, sm)); continue
        ok.append(f'    ("{name}", "{sm}", {ai}),')
    print(f"{label}: {len(ok)} valid, {len(bad)} invalid")
    for n, s in bad: print(f"    INVALID (skipped): {n} -> {s}")
    return "\n".join(ok)

acid_block = render(NEW_ACIDS, "new acid sites")
base_block = render(NEW_BASES, "new base sites")

if "boronic_acid" in src:
    print("\nalready applied - skipping edit")
else:
    # append inside each list, just before its closing bracket
    m = re.search(r"(ACID_SITES\s*=\s*\[)(.*?)(\n\])", src, re.S)
    src = src[:m.end(2)] + "\n    # --- expanded coverage ---\n" + acid_block + src[m.end(2):]
    m = re.search(r"(BASE_SITES\s*=\s*\[)(.*?)(\n\])", src, re.S)
    src = src[:m.end(2)] + "\n    # --- expanded coverage ---\n" + base_block + src[m.end(2):]
    open(path, "w", encoding="utf-8").write(src)
    print(f"\nappended to {path}")

# measure the coverage gain on real molecules
import importlib, umapka.predictor as up
importlib.reload(up)
from rdkit.Chem import PandasTools
covered = total = 0
for p2 in ["mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf",
           "mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"]:
    for mol in Chem.ForwardSDMolSupplier(p2):
        if mol is None: continue
        total += 1
        try:
            up.protonation_pair(Chem.MolToSmiles(mol)); covered += 1
        except Exception: pass
print(f"\ntest-set coverage now: {covered}/{total} ({covered/total*100:.1f}%)")
print(f"ACID_SITES: {len(up.ACID_SITES)}   BASE_SITES: {len(up.BASE_SITES)}")
