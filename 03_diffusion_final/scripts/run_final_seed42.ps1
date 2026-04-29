$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ModelDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ModelDir "..")).Path
Set-Location $ModelDir

$py = if ($env:V2V_PYTHON) { $env:V2V_PYTHON } else { "python" }
$cfg = Join-Path $ModelDir "configs\shared_loss_optuna.json"
$data = Join-Path $RepoRoot "dataset_prepared"
$root = Join-Path $ModelDir "local_runs\training_diffusion_full_seed42_v3"

if (-not (Test-Path $cfg)) { throw "shared config not found: $cfg" }
if (-not (Test-Path $data)) { throw "dataset not found: $data" }

New-Item -ItemType Directory -Path (Join-Path $root "results") -Force | Out-Null

$args = @(
    "-u", (Join-Path $ModelDir "train_diffusion.py"),
    "--dataset_root", $data,
    "--training_root", $root,
    "--epochs", "120",
    "--batch_size", "8",
    "--lr", "5e-5",
    "--min_lr", "5e-6",
    "--timesteps", "1000",
    "--sample_steps", "25",
    "--val_every", "10",
    "--warmup_epochs", "2",
    "--grad_clip", "1.0",
    "--num_workers", "0",
    "--print_every", "100",
    "--save_every", "10",
    "--shared_config", $cfg,
    "--seed", "42",
    "--amp"
)

$last = Join-Path $root "checkpoints\last_diffusion.pth"
if (Test-Path $last) {
    $args += @("--resume", $last)
}

Write-Host "final Diffusion v3 seed 42" -ForegroundColor Green
& $py $args 2>&1 | Tee-Object -FilePath (Join-Path $root "results\run_live.log")
