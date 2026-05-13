$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ModelDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ModelDir "..")).Path
Set-Location $ModelDir

$py = if ($env:V2V_PYTHON) { $env:V2V_PYTHON } else { "python" }
$cfg = Join-Path $ModelDir "configs\shared_loss_optuna.json"
$data = if ($env:V2V_DATASET_ROOT) { $env:V2V_DATASET_ROOT } else { Join-Path $RepoRoot "dataset_prepared" }
$root = Join-Path $ModelDir "local_runs\smoke_diffusion_baseline_v4_seed42"

if (-not (Test-Path $cfg)) { throw "shared config not found: $cfg" }
if (-not (Test-Path $data)) { throw "dataset not found: $data" }

New-Item -ItemType Directory -Path (Join-Path $root "results") -Force | Out-Null

$args = @(
    "-u", (Join-Path $ModelDir "train_diffusion.py"),
    "--dataset_root", $data,
    "--training_root", $root,
    "--epochs", "5",
    "--batch_size", "2",
    "--lr", "5e-5",
    "--min_lr", "5e-6",
    "--timesteps", "1000",
    "--sample_steps", "10",
    "--val_every", "1",
    "--warmup_epochs", "1",
    "--grad_clip", "1.0",
    "--max_train_steps", "20",
    "--max_eval_batches", "4",
    "--num_workers", "0",
    "--print_every", "5",
    "--save_every", "5",
    "--shared_config", $cfg,
    "--occ_bce_weight", "0.0",
    "--seed", "42",
    "--amp"
)

Write-Host "smoke Diffusion baseline v4 seed 42" -ForegroundColor Green
& $py $args 2>&1 | Tee-Object -FilePath (Join-Path $root "results\run_live.log")
