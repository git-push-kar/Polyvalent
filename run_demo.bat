@echo off
setlocal enabledelayedexpansion
title Polyvalent Unified Demo (Chat, Anomaly Detection, Finance RL)

echo ===============================================================================
echo                POLYVALENT UNIFIED MULTI-SPECIALIST DEMO
echo ===============================================================================
echo Specialists:
echo   1. General Chat / Base Model (InternVL3-2B)
echo   2. Logical Reasoning (Math / Chain-of-Thought Specialist)
echo   3. Anomaly Detection (Industrial Fault Isolation Specialist)
echo   4. Finance RL (Nifty 49 Equities Portfolio Optimization Specialist)
echo ===============================================================================
echo.

set "PY_EXE=C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY_EXE%" (
    where py >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PY_EXE=py -3.12"
    ) else (
        set "PY_EXE=python"
    )
)

echo Using Python environment: %PY_EXE%
echo Starting Unified Backend on http://127.0.0.1:8000 ...
echo Press Ctrl+C in this console to terminate the server.
echo.

start "" "http://127.0.0.1:8000"

%PY_EXE% -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1

pause
