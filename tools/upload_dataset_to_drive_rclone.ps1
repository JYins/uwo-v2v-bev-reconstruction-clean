$ErrorActionPreference = "Stop"

param(
    [string]$Remote = "gdrive",
    [string]$DriveFolder = "uwo-v2v-bev-reconstruction-dataset",
    [string]$DatasetPath = "",
    [int]$Transfers = 8,
    [int]$Checkers = 16
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $DatasetPath) {
    $DatasetPath = Join-Path $RepoRoot "dataset_prepared"
}

if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    throw "rclone was not found. Install it first, then run: rclone config"
}

if (-not (Test-Path $DatasetPath)) {
    throw "dataset folder not found: $DatasetPath"
}

$files = Get-ChildItem -Path $DatasetPath -Recurse -File -ErrorAction SilentlyContinue
$measure = $files | Measure-Object Length -Sum
$gb = [math]::Round($measure.Sum / 1GB, 2)
Write-Host "Uploading dataset_prepared: $($measure.Count) files, $gb GB" -ForegroundColor Green

$targetRoot = "${Remote}:$DriveFolder"
$target = "$targetRoot/dataset_prepared"

rclone mkdir $targetRoot

rclone copy $DatasetPath $target `
  --progress `
  --transfers $Transfers `
  --checkers $Checkers `
  --drive-chunk-size 256M `
  --retries 10 `
  --low-level-retries 20 `
  --stats 30s `
  --fast-list

Write-Host ""
Write-Host "Upload finished. Checking remote size..." -ForegroundColor Green
rclone size $target

Write-Host ""
Write-Host "Share link, if the remote supports rclone link:" -ForegroundColor Green
rclone link $targetRoot
