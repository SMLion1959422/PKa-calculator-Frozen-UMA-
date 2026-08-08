"""Fix the kind fallback: decide acid vs base by whether the atom
actually HAS a proton to lose, not by element symbol.

diagnose_v19_kind.py showed the old element-based rule scored 0/12 on
ambiguous atoms - inverted every time. Cause: ambiguous atoms are mostly
NITROGENS with an N-H (amide, sulfonamide, aromatic N-H). Those are
acidic, but "not oxygen -> base" made the code ADD a proton to an atom
that should LOSE one.

Chemistry: an atom bearing an H that a SMARTS flagged as acidic is an
acid site. An atom with a lone pair and no acidic H is a base site.
Prefers marvin_pKa_type when available, falls back to the H-count rule."""
import re
path = "eval_v19_learned_sites.py"
src = open(path, encoding="utf-8").read()

old = '''    kinds = hits[best]["kinds"]
    kind = "acid" if ("acid" in kinds and "base" not in kinds) else \\
           ("base" if ("base" in kinds and "acid" not in kinds) else
            ("acid" if nm.GetAtomWithIdx(best).GetSymbol() == "O" else "base"))
    return best, kind'''

new = '''    kinds = hits[best]["kinds"]
    if "acid" in kinds and "base" not in kinds:
        kind = "acid"
    elif "base" in kinds and "acid" not in kinds:
        kind = "base"
    else:
        # AMBIGUOUS: atom matched both acid and base patterns.
        # Decide by whether it actually HAS a proton to give up - an
        # N-H flagged as acidic (amide, sulfonamide, aromatic N-H) is
        # an acid site; a nitrogen with a lone pair and no acidic H is
        # a base site. The old element-symbol rule scored 0/12 here.
        atom = nm.GetAtomWithIdx(best)
        kind = "acid" if atom.GetTotalNumHs() > 0 else "base"
    return best, kind'''

if old in src:
    src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print("PATCHED eval_v19_learned_sites.py")
else:
    print("pattern not found - the file may already be patched")
    print("or edited; paste back the find_site() function.")
