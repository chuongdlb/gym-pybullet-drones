# Quick Conda Environment Activation Script
# Usage: .\activate-env.ps1

# Source conda hook
. C:\ProgramData\miniforge3\shell\condabin\conda-hook.ps1

# Activate the environment
conda activate gym-drones

Write-Host "✓ Conda environment activated" -ForegroundColor Green
Write-Host "Current environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor Cyan
