"""umapka - pKa prediction from UMA foundation-model embeddings."""
import importlib

__version__ = "0.1.0"
__all__ = ["PkaPredictor", "ACID_SITES", "BASE_SITES",
           "neutralize", "protonation_pair", "solvation",
           "solvents", "mixtures"]

# `predictor` needs rdkit/ase/fairchem-core; `solvation` needs nothing.
# Lazy-load so `from umapka.solvation import ...` works even without
# the heavy deps installed, and `import umapka` alone doesn't force them.
#
# IMPORTANT: use importlib.import_module here, NOT a relative
# `from . import x` statement. The latter makes Python's import
# machinery call hasattr(umapka, x) as a pre-check, which re-enters
# this __getattr__ and recurses infinitely for any name handled here
# (this bit us once already - see git history / conversation).
# importlib.import_module bypasses that hasattr pre-check entirely.
def __getattr__(name):
    if name in ("PkaPredictor", "ACID_SITES", "BASE_SITES",
                "neutralize", "protonation_pair"):
        predictor = importlib.import_module(".predictor", __name__)
        return getattr(predictor, name)
    if name == "solvation":
        return importlib.import_module(".solvation", __name__)
    if name == "solvents":
        return importlib.import_module(".solvents", __name__)
    if name == "mixtures":
        return importlib.import_module(".mixtures", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

