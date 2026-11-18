param(
 [string]$OutputDir = ""
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
 if (-not $OutputDir) { $OutputDir = 'runs/five_backbone_hybrid_screen/results_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
 if (Test-Path $OutputDir) { throw "Refusing to overwrite existing output: $OutputDir" }
 python evaluation/run_broad_hybrid_screen.py `
   --source-dir 'runs/five_backbone_hybrid_screen/source' `
   --target-dir 'runs/five_backbone_hybrid_screen/target' `
   --output-dir $OutputDir
 if ($LASTEXITCODE -ne 0) { throw 'Hybrid screen failed' }
 python evaluation/verify_broad_hybrid_screen.py $OutputDir
 if ($LASTEXITCODE -ne 0) { throw 'Verifier failed' }
 Write-Host "PASS. Results: $OutputDir"
} finally { Pop-Location }
