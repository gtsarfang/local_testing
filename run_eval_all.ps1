# Runs humaneval_bench.py against all three models in turn, unattended.
# For each: start llama-server with its tuned flags, wait for /health, run
# the eval, stop the server, wait for VRAM to actually release before the
# next one loads. Adjust $LlamaServer / $ModelsDir for your machine.
#
# Usage: powershell -File run_eval_all.ps1

$ErrorActionPreference = "Stop"
$LlamaServer = "C:\Users\George\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"
$ModelsDir = "C:\Users\George\models"

$Models = @(
    @{ Label = "qwen3-coder-30b-a3b"; File = "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"; Ncmoe = 27 },
    @{ Label = "qwen3-coder-next";    File = "Qwen3-Coder-Next-UD-Q4_K_XL.gguf";              Ncmoe = 41 },
    @{ Label = "gpt-oss-20b";         File = "gpt-oss-20b-UD-Q4_K_XL.gguf";                    Ncmoe = 4  }
)

function Wait-Health($TimeoutS = 60) {
    for ($i = 0; $i -lt $TimeoutS; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -TimeoutSec 2 -UseBasicParsing
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 1
    }
    return $false
}

foreach ($m in $Models) {
    Write-Host "`n=== $($m.Label) ===" -ForegroundColor Cyan

    $modelPath = Join-Path $ModelsDir $m.File
    $logPath = Join-Path $ModelsDir "eval_$($m.Label)_server.log"

    $proc = Start-Process -FilePath $LlamaServer -ArgumentList @(
        "-m", "`"$modelPath`"",
        "--host", "127.0.0.1", "--port", "8080",
        "-ngl", "999", "-ncmoe", $m.Ncmoe,
        "-c", "32768", "-fa", "on",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--no-mmap", "--jinja", "-np", "1"
    ) -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" -PassThru -WindowStyle Hidden

    if (-not (Wait-Health)) {
        Write-Host "  server did not become healthy in time, skipping" -ForegroundColor Red
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        continue
    }

    Write-Host "  server ready, running eval..."
    python humaneval_bench.py $m.Label

    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5   # let VRAM actually release before the next model loads
}

Write-Host "`nAll done. Results in .\results\" -ForegroundColor Green
