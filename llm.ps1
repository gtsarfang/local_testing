# One entry point for day-to-day use instead of remembering separate
# scripts. Usage:
#
#   powershell -File llm.ps1 status              # what's running, VRAM used
#   powershell -File llm.ps1 list                # available models
#   powershell -File llm.ps1 switch               # interactive menu
#   powershell -File llm.ps1 switch default        # or: next / gpt-oss
#   powershell -File llm.ps1 stop                  # stop whatever's running

param(
    [Parameter(Position = 0)]
    [ValidateSet("status", "list", "switch", "stop")]
    [string]$Command = "status",

    [Parameter(Position = 1)]
    [ValidateSet("default", "next", "gpt-oss", "gemma")]
    [string]$Model
)

$ErrorActionPreference = "Stop"

# Adjust for your machine:
$LlamaServer = "C:\Users\George\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"
$ModelsDir = "C:\Users\George\models"

# -ncmoe is the one setting that intentionally varies per model (tuned to
# leave >=~1GB free VRAM - see README "Benchmark standards" and "Tuning").
# Everything else is held fixed across all models for comparability.
$Models = [ordered]@{
    "default" = @{ File = "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"; Ncmoe = 27; Desc = "Qwen3-Coder-30B-A3B - fast, default" }
    "next"    = @{ File = "Qwen3-Coder-Next-UD-Q4_K_XL.gguf";              Ncmoe = 41; Desc = "Qwen3-Coder-Next - slower, smarter" }
    "gpt-oss" = @{ File = "gpt-oss-20b-UD-Q4_K_XL.gguf";                   Ncmoe = 4;  Desc = "gpt-oss-20b - fastest tok/s, reasoning model" }
    "gemma"   = @{ File = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf";        Ncmoe = 11; Desc = "Gemma 4 26B A4B QAT - general-purpose, not coding-specialized" }
}

function Get-RunningModelKey {
    $proc = Get-Process llama-server -ErrorAction SilentlyContinue
    if (-not $proc) { return $null }
    try {
        $props = Invoke-RestMethod -Uri "http://127.0.0.1:8080/props" -TimeoutSec 3
        $file = Split-Path $props.model_path -Leaf
        foreach ($k in $Models.Keys) {
            if ($Models[$k].File -eq $file) { return @{ Key = $k; Proc = $proc; File = $file } }
        }
        return @{ Key = $null; Proc = $proc; File = $file }
    } catch {
        return @{ Key = "loading"; Proc = $proc; File = $null }
    }
}

switch ($Command) {
    "list" {
        Write-Host "Available models:"
        foreach ($k in $Models.Keys) {
            Write-Host "  $k - $($Models[$k].Desc)"
        }
    }

    "status" {
        $running = Get-RunningModelKey
        if (-not $running) {
            Write-Host "No llama-server running." -ForegroundColor Yellow
            Write-Host "Start one with: powershell -File llm.ps1 switch"
            return
        }
        Write-Host "llama-server is running (PID $($running.Proc.Id))" -ForegroundColor Green
        if ($running.Key -eq "loading") {
            Write-Host "  still loading, not responding on :8080 yet" -ForegroundColor Yellow
            return
        }
        Write-Host "  Model: $($running.File)"
        try {
            $props = Invoke-RestMethod -Uri "http://127.0.0.1:8080/props" -TimeoutSec 3
            Write-Host "  Context: $($props.default_generation_settings.n_ctx) tokens"
        } catch {}
        $gpu = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null
        if ($gpu) { Write-Host "  VRAM: $gpu" }
    }

    "stop" {
        $proc = Get-Process llama-server -ErrorAction SilentlyContinue
        if (-not $proc) {
            Write-Host "Nothing running." -ForegroundColor Yellow
            return
        }
        Stop-Process -Id $proc.Id -Force
        Write-Host "Stopped (PID $($proc.Id))." -ForegroundColor Green
    }

    "switch" {
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

        $running = Get-RunningModelKey
        if ($running -and $running.Key -eq $Model) {
            Write-Host "$Model is already running (PID $($running.Proc.Id)) - nothing to do." -ForegroundColor Green
            return
        }
        if ($running) {
            Write-Host "Stopping current server (PID $($running.Proc.Id))..."
            Stop-Process -Id $running.Proc.Id -Force
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
                    return
                }
            } catch {}
            Start-Sleep -Seconds 1
        }
        Write-Host "Server didn't become healthy within 60s - check $logPath" -ForegroundColor Red
    }
}
