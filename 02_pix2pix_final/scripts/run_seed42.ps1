$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ModelDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ModelDir "..")).Path
Set-Location $ModelDir

$py = if ($env:V2V_PYTHON) { $env:V2V_PYTHON } else { "python" }
$cfg = Join-Path $ModelDir "configs\shared_loss_optuna.json"
$data = Join-Path $RepoRoot "dataset_prepared"
$root = Join-Path $ModelDir "local_runs\training_pix2pix_full_seed42"

if (-not (Test-Path $cfg)) { throw "shared config not found: $cfg" }
if (-not (Test-Path $data)) { throw "dataset not found: $data" }

New-Item -ItemType Directory -Path (Join-Path $root "results") -Force | Out-Null

Write-Host "final Pix2Pix seed 42, lambda_adv=0.1" -ForegroundColor Green
& $py -u (Join-Path $ModelDir "train_pix2pix.py") `
  --dataset_root $data `
  --training_root $root `
  --epochs 40 `
  --batch_size 6 `
  --g_lr 0.0002 `
  --d_lr 0.0002 `
  --lambda_adv 0.1 `
  --num_workers 0 `
  --print_every 100 `
  --save_every 10 `
  --shared_config $cfg `
  --seed 42 `
  --amp 2>&1 | Tee-Object -FilePath (Join-Path $root "results\run_live.log")
