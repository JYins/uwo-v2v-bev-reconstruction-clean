$ErrorActionPreference = "Stop"

$ModelDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ModelDir "..")).Path
Set-Location $ModelDir

$py = if ($env:V2V_PYTHON) { $env:V2V_PYTHON } else { "python" }
$data = Join-Path $RepoRoot "dataset_prepared"
$cfg = Join-Path $ModelDir "configs\shared_loss_optuna.json"
$tuneRoot = Join-Path $ModelDir "local_runs\optuna_pix2pix_adv_v2"

if (-not (Test-Path $data)) { throw "dataset not found: $data" }
if (-not (Test-Path $cfg)) { throw "shared config not found: $cfg" }
New-Item -ItemType Directory -Force -Path $tuneRoot | Out-Null

$stdout = Join-Path $tuneRoot "optuna_live.log"
$stderr = Join-Path $tuneRoot "optuna_live.err.log"

$args = @(
    "-u", (Join-Path $ModelDir "tune_pix2pix_adv.py"),
    "--dataset_root", $data,
    "--tuning_root", $tuneRoot,
    "--shared_config", $cfg,
    "--trials", "12",
    "--epochs", "8",
    "--batch_size", "6",
    "--num_workers", "0",
    "--seed", "42",
    "--prune_after", "3",
    "--adv_min", "0.1",
    "--adv_max", "2.0",
    "--adv_step", "0.1",
    "--enqueue_adv_values", "0.1,0.5,1.0,2.0",
    "--amp"
)

$p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $ModelDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Write-Host "Started Pix2Pix adversarial Optuna search."
Write-Host "PID: $($p.Id)"
Write-Host "stdout: $stdout"
Write-Host "stderr: $stderr"
