# Overnight PRELIMINARY run on the laptop (user-approved exception to "don't train on the laptop").
#   1. train baseline, then multitask (efficientnet_b0, pretrained, 256 px, batch 8, 20 epochs, early stopping)
#      out of GPU memory -> batch 4 -> batch 4 at 192 px
#   2. evaluate both on the TEST split vs pseudo-masks, bootstrap CIs -> results/preliminary/
#   3. app screenshots (3 tabs x 2 real test images) -> results/review/app/
#   4. review pack -> results/review/
# Progress: logs/overnight_status.txt  (one line per step, with time)
# Start detached (survives the Claude session):  see "Start" in logs/overnight_status.txt or run
#   powershell -NoProfile -ExecutionPolicy Bypass -File D:\fruit-grader\scripts\overnight.ps1

$ErrorActionPreference = "Continue"
$Root = "D:\fruit-grader"
Set-Location $Root
$Py = "$Root\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$Logs = "$Root\logs"
$Status = "$Logs\overnight_status.txt"
$Prelim = "results/preliminary"
$Title = "PRELIMINARY — test ground truth is auto-generated pseudo-masks, not human-verified"
New-Item -ItemType Directory -Force $Logs, "$Root\$Prelim" | Out-Null

function Say([string]$Msg) {
    $line = "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $Msg
    Add-Content -Path $Status -Value $line -Encoding UTF8
}

function Run-Py([string[]]$PyArgs, [string]$Log) {
    $p = Start-Process -FilePath $Py -ArgumentList $PyArgs -WorkingDirectory $Root -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$Log.out.log" -RedirectStandardError "$Log.err.log"
    return $p.ExitCode
}

function Train([string]$Kind) {
    $out = "runs/prelim_${Kind}_efficientnet_b0"
    foreach ($c in @(@(8, 256), @(4, 256), @(4, 192))) {
        $bs = $c[0]; $sz = $c[1]
        $log = "$Logs\train_prelim_${Kind}_bs${bs}_${sz}"
        if (Test-Path "$Root\$out") { Remove-Item -Recurse -Force "$Root\$out" }   # this run's own output only
        Say "START train $Kind (batch $bs, $sz px, 20 epochs) -> $out ; log $log.out.log"
        $t0 = Get-Date
        $code = Run-Py @("-u", "-m", "fruitgrade.train", "--model", $Kind, "--backbone", "efficientnet_b0",
                         "--epochs", "20", "--batch-size", "$bs", "--image-size", "$sz", "--out", $out) $log
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if ($code -eq 0 -and (Test-Path "$Root\$out\best.pt")) {
            $epochs = (Import-Csv "$Root\$out\history.csv").Count
            Say "DONE  train $Kind in $mins min ($epochs epochs run)"
            return @{ ok = $true; batch = $bs; image_size = $sz; out = $out; minutes = $mins; epochs_run = $epochs }
        }
        $oom = Select-String -Path "$log.err.log", "$log.out.log" -Pattern "out of memory" -Quiet
        if (-not $oom) {
            Say "FAILED train $Kind (exit $code, not out-of-memory) after $mins min - see $log.err.log"
            return @{ ok = $false; batch = $bs; image_size = $sz; out = $out; error = "exit $code" }
        }
        Say "OOM   train $Kind at batch $bs, $sz px after $mins min - retrying smaller"
    }
    Say "FAILED train $Kind - out of GPU memory at every setting"
    return @{ ok = $false; error = "out of memory at all settings" }
}

# keep the PC awake while this script runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

$started = Get-Date
$commit = (& git -C $Root rev-parse --short HEAD) 2>$null
Say "===== overnight run started (commit $commit) ====="

# 1. training
$bl = Train "baseline"
$mt = Train "multitask"

# 2. evaluation on TEST vs pseudo-masks
$evalArgs = @("-u", "-m", "fruitgrade.evaluate", "--split", "test", "--out", $Prelim, "--bootstrap", "1000",
              "--title", "`"$Title`"")
if ($mt.ok) { $evalArgs += @("--multitask", "$($mt.out)/best.pt", "--image-size", "$($mt.image_size)") }
elseif ($bl.ok) { $evalArgs += @("--image-size", "$($bl.image_size)") }
if ($bl.ok) { $evalArgs += @("--baseline", "$($bl.out)/best.pt") }
$evalOk = $false
if ($mt.ok -or $bl.ok) {
    Say "START evaluate on test (pseudo-mask ground truth, 1000 bootstrap resamples)"
    $code = Run-Py $evalArgs "$Logs\eval_prelim"
    $evalOk = ($code -eq 0)
    if ($evalOk) { Say "DONE  evaluate -> $Prelim" } else { Say "FAILED evaluate (exit $code) - see $Logs\eval_prelim.err.log" }
} else { Say "SKIP  evaluate (no model trained)" }

$settings = [ordered]@{
    title = $Title; commit = $commit; started = $started.ToString("s"); test_ground_truth = "pseudo-masks (auto-generated)"
    backbone = "efficientnet_b0"; pretrained = $true; epochs_requested = 20; early_stop_patience = "as in config.yaml"
    baseline = $bl; multitask = $mt
}
$settings | ConvertTo-Json -Depth 4 | Set-Content -Path "$Root\$Prelim\run_settings.json" -Encoding UTF8

# 3. app screenshots
if ($mt.ok -and $bl.ok) {
    Say "START app screenshots"
    $code = Run-Py @("-u", "scripts/app_screenshots.py", "--multitask", "$($mt.out)/best.pt", "--baseline", "$($bl.out)/best.pt",
                     "--report-dir", $Prelim, "--out", "results/review/app") "$Logs\app_screenshots"
    if ($code -eq 0) { Say "DONE  app screenshots -> results/review/app" } else { Say "FAILED app screenshots (exit $code) - see $Logs\app_screenshots.err.log" }
} else { Say "SKIP  app screenshots (needs both models)" }

# 4. review pack
Say "START review pack"
$code = Run-Py @("-u", "scripts/make_review.py", "--prelim", $Prelim, "--out", "results/review") "$Logs\make_review"
if ($code -eq 0) { Say "DONE  review pack -> results/review/README.md" } else { Say "FAILED review pack (exit $code) - see $Logs\make_review.err.log" }

$total = [math]::Round(((Get-Date) - $started).TotalHours, 2)
Say "===== ALL DONE in $total h ====="
[Win.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
