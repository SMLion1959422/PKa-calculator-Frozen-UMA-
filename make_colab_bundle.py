"""Package the minimum needed to run the GPU work in Google Colab.

Everything expensive on CPU here is UMA inference. PkaPredictor already
picks CUDA automatically when it is available, so no code changes are
needed - only the data and the package itself have to travel.

Deliberately EXCLUDES the big cached feature files (feat_train_v6.pkl is
127 MB, feat_atomwise.pkl grows to ~200 MB). Those are outputs of the GPU
run, not inputs; regenerating them on a GPU is faster than uploading
them.

    python make_colab_bundle.py
    -> colab_bundle.zip  (~15 MB)
"""
import os
import zipfile

FILES = [
    # the package itself
    "umapka/__init__.py",
    "umapka/predictor.py",
    "umapka/site_finder.py",
    "umapka/electronic.py",
    "umapka/microstates.py",
    "umapka/solvents.py",
    "umapka/solvation.py",
    "umapka/mixtures.py",
    "umapka/site_features.py",
    # labels: training + the two held-out sets
    "mlpka/datasets/combined_training_datasets_unique.sdf",
    "mlpka/datasets/novartis_cleaned_mono_unique_notraindata.sdf",
    "mlpka/datasets/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf",
    # models needed at inference/feature-build time
    "models/model_core_v3.pkl",          # PkaPredictor needs A regressor to load
    "models/model_core_v16_elec.pkl",     # hybrid baseline for comparison
    "models/model_core_v20_ensemble.pkl",  # current best (0.949 novartis)
    "models/site_finder_v2.pkl",          # learned site ranker
    "models/kind_classifier.pkl",         # learned acid/base kind
    "models/multisolvent_tuned.pkl",
    # the GPU jobs
    "cache_atom_embeddings.py",
    "train_attention_head.py",
    "cache_external_features.py",
    "eval_cached.py",
    "test_all.py",
    # extra labels (optional, for experiments)
    "extra_pka_data.csv",
]

OUT = "colab_bundle.zip"


def main():
    missing, total = [], 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            if not os.path.exists(f):
                missing.append(f)
                continue
            z.write(f, f)
            total += os.path.getsize(f)
    size = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}  ({size:.1f} MB compressed, {total/1e6:.1f} MB raw)")
    if missing:
        print("\nMISSING (not included):")
        for m in missing:
            print(f"  {m}")
    print("\nUpload colab_bundle.zip to Colab, then follow COLAB.md")


if __name__ == "__main__":
    main()
