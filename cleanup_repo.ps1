# Repo cleanup - run from the project root in venv311
# Organizes ~70 accumulated scripts into a publishable structure.
# Nothing is deleted; everything is COPIED into ./release/

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path release\umapka, release\scripts\pipeline,
    release\scripts\diagnostics, release\scripts\experiments, release\models,
    release\paper, release\tests | Out-Null

Write-Host "=== package ===" -ForegroundColor Cyan
Copy-Item umapka\*.py release\umapka\ -Force
Copy-Item predict_pka.py, predict_ladder.py, predict_microstates.py release\ -Force -EA SilentlyContinue

Write-Host "=== production models only ===" -ForegroundColor Cyan
# v16 = headline monoprotic model; v3 needed for the UMA embedding stack
foreach ($m in @("model_core_v16_elec.pkl","model_core_v3.pkl","model_core_v11.pkl")) {
    if (Test-Path "models\$m") { Copy-Item "models\$m" release\models\ -Force }
}

Write-Host "=== pipeline (reproduces the result) ===" -ForegroundColor Cyan
foreach ($f in @("embed_core_v6.py","compute_electronic_desc.py","train_v16_elec.py",
                 "eval_v16.py","embed_twosite.py","train_polyprotic_mlp.py",
                 "fetch_extra_pka_data.py")) {
    if (Test-Path $f) { Copy-Item $f release\scripts\pipeline\ -Force }
}

Write-Host "=== diagnostics (the audit trail) ===" -ForegroundColor Cyan
foreach ($f in @("audit_data_quality.py","diagnose_site_mismatch.py","verify_marvin_atoms.py",
                 "check_test_novelty.py","check_contamination.py","check_marvin_baseline.py",
                 "check_methodology_holes.py","build_polyprotic_benchmark.py",
                 "ablation_uma_vs_electronic.py","ablation_xtb.py")) {
    if (Test-Path $f) { Copy-Item $f release\scripts\diagnostics\ -Force }
}

Write-Host "=== tests ===" -ForegroundColor Cyan
foreach ($f in @("test_smarts_coverage.py","test_site_tagging.py","test_v4_functions.py")) {
    if (Test-Path $f) { Copy-Item $f release\tests\ -Force }
}

Write-Host "=== negative results (keep - they are in the paper) ===" -ForegroundColor Cyan
foreach ($f in @("train_site_selector.py","eval_core_v8.py","train_split_acid_base.py",
                 "train_split_full.py","train_core_v3_sizeweighted.py","pilot_relaxed_geometry.py",
                 "pilot_bestconf.py","compute_xtb_desc.py","diagnose_xtb_failures.py")) {
    if (Test-Path $f) { Copy-Item $f release\scripts\experiments\ -Force }
}

Write-Host "=== paper ===" -ForegroundColor Cyan
foreach ($f in @("URTC_paper.pdf","URTC_paper.tex","fig1.pdf")) {
    if (Test-Path $f) { Copy-Item $f release\paper\ -Force }
}

# .gitignore - keep the big caches out of git
@'
*.pkl.partial
feat_*.pkl
extra_pka_source/
venv311/
__pycache__/
*.pyc
characterization_*.csv
core_maxdata.pkl
'@ | Set-Content release\.gitignore -Encoding utf8

# minimal README
@'
# umapka

pKa prediction from frozen Meta UMA foundation-model embeddings combined
with classical electronic descriptors.

## Results

| benchmark | this work | ChemAxon Marvin |
|---|---|---|
| Novartis (n=280) | **0.845** | 0.856 |
| AvLiLuMoVe (n=122) | **0.441** | 0.566 |

MAE in pKa units. Marvin values computed from annotations distributed with
the same SDF files, on identical molecules.

## Quick start

```
pip install -r requirements.txt
python predict_pka.py "CC(=O)Oc1ccccc1C(=O)O"
python predict_microstates.py "OC(=O)CC(N)C(=O)O"   # polyprotic
```

## Layout

- `umapka/` - the package
- `scripts/pipeline/` - reproduces the trained model end to end
- `scripts/diagnostics/` - data audits, leakage checks, ablations
- `scripts/experiments/` - approaches that were tested and rejected
- `tests/` - correctness tests (SMARTS coverage, atom-index tracking)
- `paper/` - manuscript

## Known limitations

- Reliable for pKa 2-12; training contains no examples outside that range
- Polyprotic MAE is 2.0 on a 30-molecule benchmark - not yet production quality
- Headline numbers use ChemAxon Marvin site annotations at inference;
  self-contained site finding costs ~0.15 pKa units
- See the paper for four documented negative results

## License

Apache-2.0
'@ | Set-Content release\README.md -Encoding utf8

@'
rdkit
numpy
pandas
scikit-learn
lightgbm
joblib
tqdm
torch
fairchem-core
ase
'@ | Set-Content release\requirements.txt -Encoding utf8

Write-Host "`n=== done ===" -ForegroundColor Green
Get-ChildItem -Recurse release -File | Group-Object DirectoryName |
    ForEach-Object { Write-Host ("  {0,-45} {1} files" -f $_.Name.Replace((Get-Location).Path,'.'), $_.Count) }
$sz = (Get-ChildItem -Recurse release -File | Measure-Object Length -Sum).Sum/1MB
Write-Host ("`n  total: {0:N1} MB" -f $sz)
Write-Host "`nTo publish:" -ForegroundColor Cyan
Write-Host "  cd release; git init; git add .; git commit -m 'initial release'"
