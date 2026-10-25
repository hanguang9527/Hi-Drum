# Hi-Drum environment setup (Ubuntu)

Guide for recreating the `drum` Conda environment used for Hi-Drum training and evaluation.

## 1. Prerequisites

- OS: Ubuntu 20.04/22.04 (or a compatible Linux distro)
- Anaconda or Miniconda installed
- Network access (the bootstrap script optionally uses a Tsinghua PyPI mirror)

## 2. Files

Keep these two files in the repo root:

1. `requirements_ubuntu.txt` — pinned dependencies (Python 3.10.19 / `drum` env)
2. `deploy_env.sh` — automated bootstrap script

## 3. One-shot setup

```bash
cd /path/to/Hi-Drum
chmod +x deploy_env.sh
./deploy_env.sh
```

The script creates a Conda env named `drum` and installs the locked requirements. This may take several minutes.

## 4. Activate and verify

```bash
conda activate drum
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 5. FAQ

**Q: `conda: command not found`**  
Install Miniconda/Anaconda, or run `source ~/.bashrc` after install.

**Q: How do I rename the env?**  
Edit `ENV_NAME="drum"` near the top of `deploy_env.sh`, then re-run the script.

**Q: Slow or timing-out package downloads**  
The script defaults to a Tsinghua mirror. On overseas hosts, remove the `-i https://pypi.tuna.tsinghua.edu.cn/simple` flag from `deploy_env.sh`.
