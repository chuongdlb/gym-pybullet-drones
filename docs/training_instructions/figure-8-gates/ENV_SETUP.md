# Environment Setup Quick Reference

## Conda Environment Activation

### Quick Activation (PowerShell)
```powershell
.\activate-env.ps1
```

### Manual Activation
```powershell
# Source conda hook
. C:\ProgramData\miniforge3\shell\condabin\conda-hook.ps1

# Activate environment
conda activate gym-pybullet-drones
```

### Create New Environment (if needed)
```powershell
conda create -n gym-pybullet-drones python=3.8 -y
conda activate gym-pybullet-drones
pip install -e .
```

### Verify Environment
```powershell
python verify_env.py
```

## Common Commands

### Run Examples
```powershell
python gym_pybullet_drones/examples/pid.py
python gym_pybullet_drones/examples/learn_gates.py
```

### Run Tests
```powershell
python test_pid.py
```
