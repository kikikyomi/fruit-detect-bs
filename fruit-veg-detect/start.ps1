$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$BackendPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
$FrontendUrl = "http://127.0.0.1:5173"

function Test-PortListening {
    param([int]$Port)

    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    return $null -ne $connection
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Message
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw $Message
    }
}

function Start-CommandWindow {
    param(
        [string]$Title,
        [string]$WorkingDirectory,
        [string]$Command
    )

    $cmd = "title $Title && cd /d `"$WorkingDirectory`" && $Command"
    Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $cmd)
}

Write-Host "Fruit Veg Detect one-click startup"
Write-Host "Project: $RootDir"
Write-Host ""

Assert-PathExists -Path $BackendDir -Message "Backend directory not found: $BackendDir"
Assert-PathExists -Path $FrontendDir -Message "Frontend directory not found: $FrontendDir"
Assert-PathExists -Path $BackendPython -Message "Backend virtualenv python not found: $BackendPython"

if (Test-PortListening -Port 8000) {
    Write-Host "Backend is already listening on 127.0.0.1:8000"
} else {
    Write-Host "Starting backend on 127.0.0.1:8000 ..."
    Start-CommandWindow `
        -Title "Fruit Veg Backend" `
        -WorkingDirectory $BackendDir `
        -Command "`"$BackendPython`" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
}

if (Test-PortListening -Port 5173) {
    Write-Host "Frontend is already listening on 127.0.0.1:5173"
} else {
    Write-Host "Starting frontend on 127.0.0.1:5173 ..."
    Start-CommandWindow `
        -Title "Fruit Veg Frontend" `
        -WorkingDirectory $FrontendDir `
        -Command "npm run dev -- --host 127.0.0.1 --port 5173"
}

Write-Host ""
Write-Host "Opening $FrontendUrl ..."
Start-Sleep -Seconds 3
Start-Process $FrontendUrl

Write-Host ""
Write-Host "Done. Keep the backend/frontend command windows open while using the system."
