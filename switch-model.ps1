# Switches the running llama-server to a different model, without having to
# remember which start-*.ps1 script goes with which model or manually stop
# the old one first.
#
# Usage:
#   powershell -File switch-model.ps1                    # interactive menu
#   powershell -File switch-model.ps1 -Model default      # direct
#   powershell -File switch-model.ps1 -Model next
#   powershell -File switch-model.ps1 -Model gpt-oss

param(
    [ValidateSet("default", "next", "gpt-oss")]
    [string]$Model
)

$ErrorActionPreference = "Stop"

# Adjust for your machine (matches the individual start-*.ps1 scripts):
$LlamaServer = "C:\Users\George\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"
$ModelsDir = "C:\Users\George\models"

$Models = @{
    "default" = @{ File = "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"; Ncmoe = 27; Desc = "Qwen3-Coder-30B-A3B - fast, default" }
    "next"    = @{ File = "Qwen3-Coder-Next-UD-Q4_K_XL.gguf";              Ncmoe = 41; Desc = "Qwen3-Coder-Next - slower, smarter" }
    "gpt-oss" = @{ File = "gpt-oss-20b-UD-Q4_K_XL.gguf";                   Ncmoe = 4;  Desc = "gpt-oss-20b - fastest tok/s, reasoning model" }
}

if (-not $Model) {
    Write-Host "Which model?"
    $i = 1
    $keys = @($Models.Keys)
    foreach ($k in $keys) {
        Write-Host "  [$i] $k - $($Models[$k].Desc)"
        $i++
    }
    $choice = Read-Host "Enter number"
    $idx = [int]$choice - 1
    if ($idx -lt 0 -or $idx -ge $keys.Count) {
        Write-Host "Invalid choice." -ForegroundColor Red
        exit 1
    }
    $Model = $keys[$idx]
}

$cfg = $Models[$Model]
$modelPath = Join-Path $ModelsDir $cfg.File

# Skip the restart if the right model's already loaded.
$existing = Get-Process llama-server -ErrorAction SilentlyContinue
if ($existing) {
    try {
        $props = Invoke-RestMethod -Uri "http://127.0.0.1:8080/props" -TimeoutSec 3
        if ((Split-Path $props.model_path -Leaf) -eq $cfg.File) {
            Write-Host "$Model is already running (PID $($existing.Id)) - nothing to do." -ForegroundColor Green
            exit
        }
    } catch {}
    Write-Host "Stopping current server (PID $($existing.Id))..."
    Stop-Process -Id $existing.Id -Force
    Start-Sleep -Seconds 5   # let VRAM actually release before the next model loads
}

Write-Host "Starting $Model ($($cfg.Desc))..."
$logPath = Join-Path $ModelsDir "server.log"
Start-Process -FilePath $LlamaServer -ArgumentList @(
    "-m", "`"$modelPath`"",
    "--host", "127.0.0.1", "--port", "8080",
    "-ngl", "999", "-ncmoe", $cfg.Ncmoe,
    "-c", "32768", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-mmap", "--jinja", "-np", "1"
) -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" -WindowStyle Hidden

for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            Write-Host "$Model is ready." -ForegroundColor Green
            exit
        }
    } catch {}
    Start-Sleep -Seconds 1
}
Write-Host "Server didn't become healthy within 60s - check $logPath" -ForegroundColor Red
