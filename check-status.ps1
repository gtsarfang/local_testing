# Quick status check: is llama-server running, and which model is loaded?
# Usage: powershell -File check-status.ps1

$proc = Get-Process llama-server -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "No llama-server running." -ForegroundColor Yellow
    Write-Host "Start one with: powershell -File switch-model.ps1"
    exit
}

Write-Host "llama-server is running (PID $($proc.Id))" -ForegroundColor Green

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -TimeoutSec 3
    if ($health.status -ne "ok") {
        Write-Host "  /health returned unexpected status: $($health.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  Process is running but not responding on :8080 yet (still loading?)" -ForegroundColor Yellow
    exit
}

try {
    $props = Invoke-RestMethod -Uri "http://127.0.0.1:8080/props" -TimeoutSec 3
    $modelFile = Split-Path $props.model_path -Leaf
    Write-Host "  Model: $modelFile"
    Write-Host "  Context: $($props.default_generation_settings.n_ctx) tokens"
} catch {
    Write-Host "  Could not read /props (older build?)" -ForegroundColor Yellow
}

$gpu = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null
if ($gpu) {
    Write-Host "  VRAM: $gpu"
}
