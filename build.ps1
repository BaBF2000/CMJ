# ==========================================
# CMJ Framework - Build script
# ==========================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "====================================="
Write-Host " Building CMJ Framework executable"
Write-Host "====================================="
Write-Host ""

# ----------------------------
# Resolve project root
# ----------------------------
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"

# ----------------------------
# Virtual environment
# ----------------------------
$PythonExe = Join-Path $ProjectRoot ".venv_build\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python executable not found in .venv_build" -ForegroundColor Red
    Write-Host "Expected: $PythonExe" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $PythonExe"

# ----------------------------
# Clean previous builds
# ----------------------------
Write-Host ""
Write-Host "Cleaning previous build folders..."

if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}

if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}

# Optional: clear old spec cache folders created by PyInstaller
if (Test-Path "__pycache__") {
    Remove-Item -Recurse -Force "__pycache__"
}

# ----------------------------
# Regenerate documentation HTML
# ----------------------------
Write-Host ""
Write-Host "Updating documentation HTML..."

& $PythonExe -m jupyter nbconvert `
    --to html `
    documentation/CMJ_Framework_Documentation.ipynb `
    --output CMJ_Framework_Documentation.html `
    --output-dir documentation

# ----------------------------
# Run PyInstaller
# ----------------------------
Write-Host ""
Write-Host "Running PyInstaller..."

& $PythonExe -m PyInstaller `
    --clean `
    --noconfirm `
    CMJ_Manager.spec

# ----------------------------
# Final checks
# ----------------------------
$ExeFolder = Join-Path $ProjectRoot "dist\CMJ_Manager"
$ExePath = Join-Path $ExeFolder "CMJ_Manager.exe"

Write-Host ""
if (Test-Path $ExePath) {
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host " Build completed successfully" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "Executable: $ExePath" -ForegroundColor Green
} else {
    Write-Host "WARNING: Build finished, but executable was not found at:" -ForegroundColor Yellow
    Write-Host $ExePath -ForegroundColor Yellow
}