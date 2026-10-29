#!/bin/bash

###############################################################################
# Megatron-Bridge runtime environment
#
# Loads an already-installed Megatron-Bridge environment.
# Does NOT create, install, clone, checkout or modify anything.
#
# Known working configuration:
#   Python 3.12.3
#   CUDA 13.2
#   PyTorch 2.13.0 (+ CUDA 13.0)
#   Transformer Engine 2.18.0
#   Megatron-Bridge 846216ed4704283a249c19c7b6d93622c5bd9f37 (baseline commit used to create the repository and environment; the current commit is expected to differ as development progresses)
#   Megatron-Core 2a75ac12c54ba6a42b34024756820bf819a2e25f
#   NVIDIA Resiliency Extension 6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d
#   Apex a1d527a857e8da64c4e7237ca89ec699fb4d9eaf
###############################################################################

###############################################################################
# 1. Configuration
###############################################################################

# We load COPLA project environment variables
source /gpfs/projects/bsc88/copla/copla.env

# You SHOULD check these variables and change them if necessary
USER_NAME="$(whoami)"
ENV_NAME="pre-training_megatron-bridge_mn5_python3.12.3_20261027"

ENV_ROOT="${COPLA_PROJECTS}/environments/${ENV_NAME}"
HF_CACHE="/gpfs/scratch/bsc88/${USER_NAME}/cache/hf_cache"

export BRIDGE_ROOT="/gpfs/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge"
export NVRX_ROOT="${COPLA_CODEBASE}/nvidia-resiliency-ext-0.6.0"
export APEX_ROOT="${COPLA_CODEBASE}/apex"

CUDA_ROOT="/apps/ACC/CUDA/13.2"
export MODULEPATH=$MODULEPATH:/apps/GPP/EASYBUILD/modules/all

###############################################################################
# 2. Modules
###############################################################################

module purge

module load EB/apps
module load Python/3.12.3-GCCcore-13.3.0
module load intel-compilers/2024.2.0
module load cuda/13.2

###############################################################################
# 3. Activate virtual environment
###############################################################################

if [ ! -d "${ENV_ROOT}" ]; then
    echo "ERROR: virtual environment does not exist:"
    echo "       ${ENV_ROOT}"
    echo
    echo "Run the installation script first."
    return 1 2>/dev/null || exit 1
fi

source "${ENV_ROOT}/bin/activate"

# Ensure Python imports Megatron-Bridge and Megatron-Core from the current user's local clone instead of the shared virtual environment's editable-install paths.
export PYTHONPATH="${BRIDGE_ROOT}/src:${BRIDGE_ROOT}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

###############################################################################
# 4. CUDA environment
###############################################################################

export CUDA_HOME="${CUDA_ROOT}"
export CUDA_INC="${CUDA_ROOT}/include"
export CUDA_INSTALL_PATH="${CUDA_ROOT}"
export CUDA_VERSION="13.2"

###############################################################################
# 5. Hugging Face cache / offline mode
###############################################################################

export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"

export TRANSFORMERS_OFFLINE=1

export WANDB_MODE=offline

###############################################################################
# 6. CUDA / cuDNN paths from the Python environment
###############################################################################

export CUDNN_PATH="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn"
export CUDNN_INCLUDE_DIR="${CUDNN_PATH}/include"
export CUDNN_LIB_DIR="${CUDNN_PATH}/lib"
export CUDNN_LIBRARY="${CUDNN_PATH}/lib"

export NVTE_CUDA_INCLUDE_DIR="${CUDA_ROOT}/include"

###############################################################################
# 7. Compiler include paths
###############################################################################

export C_INCLUDE_PATH="${CUDNN_INCLUDE_DIR}:${CUDA_ROOT}/extras/CUPTI/include:${CUDA_ROOT}/nvvm/include:${CUDA_ROOT}/include"

export CPLUS_INCLUDE_PATH="${CUDNN_INCLUDE_DIR}:${CUDA_ROOT}/extras/CUPTI/include:${CUDA_ROOT}/nvvm/include:${CUDA_ROOT}/include"

###############################################################################
# 8. Runtime library paths
###############################################################################

export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cu13/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

###############################################################################
# 9. Move to Megatron-Bridge
###############################################################################

cd "${BRIDGE_ROOT}"

###############################################################################
# 10. Verification
###############################################################################

echo
echo "Testing imports, repository paths, and Git commits..."

python - <<'PY' 
import os
import sys
import subprocess
from pathlib import Path

import torch
import transformer_engine
import megatron.bridge
import megatron.core
import megatron.training
import apex
import fused_weight_gradient_mlp_cuda

bridge_root = Path(os.environ["BRIDGE_ROOT"]).resolve()
nvrx_root = Path(os.environ["NVRX_ROOT"]).resolve()
apex_root = Path(os.environ["APEX_ROOT"]).resolve()

expected_paths = {
    "megatron.bridge": bridge_root / "src" / "megatron" / "bridge",
    "megatron.core": bridge_root / "3rdparty" / "Megatron-LM" / "megatron" / "core",
    "megatron.training": bridge_root / "3rdparty" / "Megatron-LM" / "megatron" / "training",
}

modules = {
    "megatron.bridge": megatron.bridge,
    "megatron.core": megatron.core,
    "megatron.training": megatron.training,
}

expected_commits = {
    "Megatron-Core": (
        bridge_root / "3rdparty" / "Megatron-LM",
        "2a75ac12c54ba6a42b34024756820bf819a2e25f",
    ),
    "NVIDIA Resiliency Extension": (
        nvrx_root,
        "6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d",
    ),
    "Apex": (
        apex_root,
        "a1d527a857e8da64c4e7237ca89ec699fb4d9eaf",
    ),
}

errors = []

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Transformer Engine:", transformer_engine.__version__)
print("Apex import: OK")
print("fused_weight_gradient_mlp_cuda import: OK")

print("\nImport paths:")
for name, module in modules.items():
    actual = Path(module.__file__).resolve()
    expected = expected_paths[name].resolve()

    print(f"{name}: {actual}")

    if not actual.is_relative_to(expected):
        errors.append(f"{name} is not loaded from the current user's repository")

print("\nGit commit verification:")
for name, (repo, expected_commit) in expected_commits.items():
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

        print(f"{name}: {actual_commit}")

        if actual_commit != expected_commit:
            errors.append(
                f"{name}: expected {expected_commit}, got {actual_commit}"
            )
    except subprocess.CalledProcessError as exc:
        errors.append(f"{name}: unable to read Git commit: {exc.output.strip()}")

if errors:
    print("\nERROR: Verification failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("\nAll imports, paths, and Git commits verified successfully.")
PY

status=$?
if [ "$status" -ne 0 ]; then
    echo "ERROR: Environment verification failed."
    return "$status"
fi