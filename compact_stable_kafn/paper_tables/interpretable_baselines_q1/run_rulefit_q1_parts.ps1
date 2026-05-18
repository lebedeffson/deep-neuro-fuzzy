Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location "D:\Sasha\deep-neuro-fuzzy"
$py = ".\.venv_run\Scripts\python.exe"
$outDir = "compact_stable_kafn/paper_tables/interpretable_baselines_q1"
$datasets = @("breast_cancer", "susy_binary_200000")
$budgets = @(100, 200, 400)
$seeds = "7,19,23,29,42,101"

foreach ($ds in $datasets) {
  foreach ($b in $budgets) {
    Write-Output "[part] dataset=$ds budget=$b start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    & $py "compact_stable_kafn/scripts/run_interpretable_baselines.py" `
      --datasets $ds `
      --seeds $seeds `
      --models rulefit `
      --rulefit-budgets "$b" `
      --resume `
      --checkpoint-every-run `
      --out-dir $outDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Output "[part] dataset=$ds budget=$b done  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  }
}

Write-Output "[ok] all parts finished"
