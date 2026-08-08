$lines = @(
    'import sys, sklearn, os',
    'if "venv311" not in sys.prefix:',
    '    sys.exit(f"WRONG PYTHON: {sys.prefix}\n  activate venv311 first: .\\venv311\\Scripts\\Activate.ps1")'
)
foreach ($f in Get-ChildItem -Filter "predict_*.py") {
    $c = Get-Content $f.FullName -Raw
    if (-not $c.Contains("WRONG PYTHON")) {
        $guard = ($lines -join "`n") + "`n`n"
        Set-Content -Path $f.FullName -Value ($guard + $c) -Encoding utf8
        Write-Host "guarded $($f.Name)" -ForegroundColor Green
    }
}
