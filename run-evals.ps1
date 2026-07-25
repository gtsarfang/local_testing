# Runs an eval against one or more models in turn, unattended. Replaces the
# separate run_eval_all.ps1 / run_lcb_eval_all.ps1 (they duplicated the same
# start/health-poll/stop logic that llm.ps1 already does) - this calls
# llm.ps1 switch for server management instead of reimplementing it.
#
# Usage:
#   powershell -File run-evals.ps1 humaneval                        # all models
#   powershell -File run-evals.ps1 humaneval default,next            # subset
#   powershell -File run-evals.ps1 livecodebench gemma

param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("humaneval", "livecodebench")]
    [string]$Eval,

    [Parameter(Position = 1)]
    [string[]]$Models = @("default", "next", "gpt-oss", "gemma")
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot

# Maps llm.ps1's short model keys to the labels already used in committed
# results/*.json filenames, so existing history stays intact.
$Labels = @{
    "default" = "qwen3-coder-30b-a3b"
    "next"    = "qwen3-coder-next"
    "gpt-oss" = "gpt-oss-20b"
    "gemma"   = "gemma-4-26b-a4b"
}
$EvalScript = if ($Eval -eq "humaneval") { "humaneval_bench.py" } else { "livecodebench_bench.py" }

foreach ($m in $Models) {
    $label = $Labels[$m]
    Write-Host "`n=== $m ($label) ===" -ForegroundColor Cyan

    & powershell -NoProfile -File (Join-Path $ScriptDir "llm.ps1") switch $m

    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -ne 200) { throw "not healthy" }
    } catch {
        Write-Host "  $m did not come up healthy, skipping" -ForegroundColor Red
        continue
    }

    Write-Host "  running $Eval eval..."
    python (Join-Path $ScriptDir $EvalScript) $label
}

& powershell -NoProfile -File (Join-Path $ScriptDir "llm.ps1") stop
Write-Host "`nAll done. Results in .\results\" -ForegroundColor Green
