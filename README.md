# Project Maji

## Road To SKA Project H

Mapping water in Africa using satellite data

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
```

### 2. Install PyTorch (choose your platform)

```bash
# macOS Apple Silicon (M1/M2/M3) - MPS acceleration
pip install torch torchvision

# Linux with NVIDIA GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Linux with NVIDIA GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (any platform)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```
