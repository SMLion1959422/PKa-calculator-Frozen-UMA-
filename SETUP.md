# Setup

This project depends on PyTorch (via `fairchem-core`) and a gated
Hugging Face model, so a fresh clone needs a few explicit steps beyond
`pip install -e .`. Follow these in order on **every machine** you set
this up on.

## 0. Requirements

- Python **3.11** (the version this project is developed and tested with)
- ~2 GB free disk (UMA weights + dependencies)
- A [Hugging Face](https://huggingface.co) account
- NVIDIA GPU recommended (~0.35 s/molecule) but not required -- CPU works,
  just slower (several seconds/molecule)

## 1. Clone and create a virtual environment

```bash
git clone https://github.com/SMLion1959422/umapka.git
cd umapka
python3.11 -m venv venv311
```

Activate it:
- **macOS/Linux:** `source venv311/bin/activate`
- **Windows PowerShell:** `venv311\Scripts\Activate.ps1`
- **Windows cmd.exe:** `venv311\Scripts\activate.bat`

Your prompt should now start with `(venv311)`. Every command below
assumes this is active -- reactivate it any time you open a new terminal.

```bash
python -m pip install --upgrade pip
```

## 2. Install PyTorch first, explicitly

Don't let `pip install fairchem-core` pick a torch build for you --
install it yourself so it matches your hardware.

**CPU only (any OS, including Apple Silicon):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (Linux/Windows)** -- check
[pytorch.org](https://pytorch.org/get-started/locally/) for the exact
command matching your driver's CUDA version, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Already have `requirements-lock.txt` from a working machine with the
SAME CPU/GPU profile as this one?** Skip straight to step 3's lock-file
command instead of picking an index-url by hand -- see "Reproducing on
additional machines" below for why it only works when the profile matches.

## 3. Install the project

For a first install on a new hardware profile:
```bash
pip install -e .
```

For a repeat install on a machine with the **same CPU/GPU profile** as
the one `requirements-lock.txt` was frozen on:
```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```
(`--no-deps` avoids re-resolving anything the lock file already pinned.)

## 4. Get access to the gated UMA model

1. Request access at https://huggingface.co/facebook/UMA (one-time per
   HF account; approval can take anywhere from instant to a day or two).
2. Create a token (read access is enough) at
   https://huggingface.co/settings/tokens.
3. Log in **on this machine**:
```bash
pip install -U huggingface_hub
hf auth login
```
   (`huggingface-cli login` still works but is deprecated in favor of
   `hf auth login`.)

Confirm:
```bash
hf auth whoami
```

## 5. Always run from the repo root

`predict_pka.py` and the examples use relative paths to the `models/`
folder. Running from any other working directory will throw a
`FileNotFoundError` that looks like a broken install but isn't -- just
`cd` back to the repo root.

## 6. Smoke test

```bash
python -c "import torch, rdkit, ase, fairchem.core; print('imports OK')"
hf auth whoami
python examples/quickstart.py
python predict_pka.py "CC(=O)O"
python predict_pka.py "CC(=O)O" --solvent dmso
```
If all of these succeed, the environment is fully working -- imports,
Hugging Face auth, water prediction, and non-water solvent prediction.

## Known platform notes

- **Windows:** LightGBM can throw a memory access violation on
  non-contiguous NumPy arrays. `umapka/predictor.py` already forces
  contiguous `float64` arrays before every regressor call to avoid this
  -- no action needed, just don't remove that conversion if editing that
  file.
- **Apple Silicon:** the code only checks for `cuda`, so it always falls
  back to CPU on Mac (no crash, just slower -- no `mps` support here yet).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401`/`403` downloading the model | Not logged into HF, or access not yet approved | Step 4 |
| `FileNotFoundError: models/...pkl` | Running from the wrong directory | Step 5 |
| Import errors mentioning CUDA | Wrong torch build for this machine, or used the lock file on a mismatched profile | Redo step 2 fresh (don't use the lock file across different CPU/GPU profiles) |
| `UnboundLocalError` involving `np` | Old bug in `_base_pka`, already fixed | Make sure you're on the current `predictor.py` |
| Very slow (seconds/molecule) | Expected on CPU/Mac | Not a bug -- use a CUDA machine for speed |
| `venv311\Scripts\Activate.ps1` blocked | Windows execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry |

## Reproducing on additional machines

Steps 1, 4's login, and 5 must be repeated on every machine; the HF
account/access request in step 4 is one-time.

`requirements-lock.txt` in this repo was frozen from a known-working
install. **It only reproduces correctly on a machine with the same
CPU/GPU profile** (CUDA version, or CPU-only) as the one it came from --
`torch`'s own version string bakes in which build was installed (e.g.
`+cu121` vs `+cpu`), so the lock file can't bridge across different
hardware profiles by itself:

- **Same profile as the lock file:** skip step 2 entirely, go straight
  to step 3's `pip install -r requirements-lock.txt` command.
- **Different profile (e.g. this machine has no GPU but the lock file
  was frozen on a CUDA machine, or vice versa):** do NOT use the lock
  file -- follow steps 2-3's fresh-install path instead. Once that's
  working, you can regenerate a profile-specific lock file with
  `pip freeze > requirements-lock-<profile>.txt` (e.g.
  `requirements-lock-cpu.txt`, `requirements-lock-cuda121.txt`) if you
  want a matching one for future machines of that same type.
