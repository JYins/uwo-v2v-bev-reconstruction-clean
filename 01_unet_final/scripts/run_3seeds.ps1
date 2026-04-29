$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ModelDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ModelDir "..")).Path
Set-Location $ModelDir

$py = if ($env:V2V_PYTHON) { $env:V2V_PYTHON } else { "python" }
$cfg = Join-Path $ModelDir "configs\shared_loss_optuna.json"
$data = Join-Path $RepoRoot "dataset_prepared"
$seeds = @(42, 43, 44)

if (-not (Test-Path $cfg)) { throw "shared config not found: $cfg" }
if (-not (Test-Path $data)) { throw "dataset not found: $data" }

foreach ($seed in $seeds) {
    $root = Join-Path $ModelDir "local_runs\training_unet_optuna_seed$seed"
    New-Item -ItemType Directory -Path (Join-Path $root "results") -Force | Out-Null

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] final U-Net seed $seed" -ForegroundColor Green

    & $py -u (Join-Path $ModelDir "train.py") `
      --dataset_root $data `
      --training_root $root `
      --epochs 80 `
      --batch_size 12 `
      --features 16,32,64,128 `
      --num_workers 0 `
      --print_every 100 `
      --save_every 10 `
      --shared_config $cfg `
      --seed $seed 2>&1 | Tee-Object -FilePath (Join-Path $root "results\run_live.log")
}
