# Polyvalent Unified Demo Launcher
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "               POLYVALENT UNIFIED MULTI-SPECIALIST DEMO" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "Specialists: Chat (General), Reasoning, Anomaly Detection, Finance RL" -ForegroundColor Gray
Write-Host ""

$pyExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $pyExe)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pyExe = "py -3.12"
    } else {
        $pyExe = "python"
    }
}

Set-Location -LiteralPath $PSScriptRoot
# This demo machine has room for the shared base and forensic wrapper together.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuMemory = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1
    if ([int]$gpuMemory -ge 24000) {
        $env:FINAL_ADAPTER_POLICY = 'keep'
        $env:FINAL_LAYA_ROUTER_DEVICE = 'cuda'
    }
}
Write-Host "Using Python: $pyExe" -ForegroundColor Cyan
Write-Host "Launching web browser at http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"

if ($pyExe -like "* *") {
    & py -3.12 -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1
} else {
    & $pyExe -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1
}
