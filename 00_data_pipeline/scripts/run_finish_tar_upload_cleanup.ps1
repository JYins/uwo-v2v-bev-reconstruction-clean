$py = "D:\Anaconda3\envs\v2v4real\python.exe"
$script = "D:\MEng_Project\finish_tar_upload_cleanup.py"
$archive = "C:\narval_transfer\dataset_prepared.tar"
$logOut = "D:\MEng_Project\narval_sync_logs\finish_tar_upload_stdout.log"
$logErr = "D:\MEng_Project\narval_sync_logs\finish_tar_upload_stderr.log"

New-Item -ItemType Directory -Force -Path "D:\MEng_Project\narval_sync_logs" | Out-Null

$args = @(
  "-u",
  $script,
  "--archive", $archive,
  "--remote_tar", "/home/syin94/scratch/MEng_Project/data/dataset_prepared.tar",
  "--remote_extract_dir", "/home/syin94/scratch/MEng_Project/data",
  "--user", "syin94",
  "--password", "Yin1225s*",
  "--duo_code", "036777190",
  "--key_path", "C:\Users\shi\.ssh\id_ed25519_alliance",
  "--poll_seconds", "20",
  "--stable_polls", "3",
  "--delete_remote_tar"
)

Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
