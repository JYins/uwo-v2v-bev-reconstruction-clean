$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ModelDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ModelDir "..")).Path
Set-Location $ModelDir

$py = if ($env:V2V_PYTHON) { $env:V2V_PYTHON } else { "python" }
$data = Join-Path $RepoRoot "dataset_prepared"
$tuneRoot = Join-Path $ModelDir "local_runs\optuna_unet_run"

if (-not (Test-Path $data)) { throw "dataset not found: $data" }
New-Item -ItemType Directory -Path $tuneRoot -Force | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[$stamp] U-Net Optuna loss search" -ForegroundColor Green

& $py -u (Join-Path $ModelDir "tune_unet_optuna.py") `
  --dataset_root $data `
  --tuning_root $tuneRoot `
  --trials 24 `
  --epochs 12 `
  --batch_size 12 `
  --num_workers 0 `
  --print_every 300 `
  --prune_after 4 `
  --seed 42 2>&1 | Tee-Object -FilePath (Join-Path $tuneRoot "optuna_live.log") -Append
