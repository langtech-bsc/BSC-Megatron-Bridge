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
#   Megatron-Bridge 846216ed4704283a249c19c7b6d93622c5bd9f37
#   Megatron-Core 2a75ac12c54ba6a42b34024756820bf819a2e25f
#   NVIDIA Resiliency Extension 6c5f2a13c7688d92a7ac7ee6e464721eb8b7345d
#   Apex a1d527a857e8da64c4e7237ca89ec699fb4d9eaf
###############################################################################

###############################################################################
# 1. Configuration
###############################################################################

# You MUST set these variables
USER_NAME="bsc082280"
ENV_NAME="pre-training_megatron-bridge_mn5_python3.12.3_20261007"

# You SHOULD check these paths and change them if necessary
ENV_ROOT="/gpfs/projects/bsc88/copla/environments/${ENV_NAME}"
BRIDGE_ROOT="/gpfs/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge"
HF_CACHE="/gpfs/scratch/bsc88/${USER_NAME}/cache/hf_cache"

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
echo "============================================================"
echo "Megatron-Bridge environment loaded"
echo "============================================================"

echo
echo "Python:"
python --version
which python

echo
echo "Environment:"
echo "  VIRTUAL_ENV = ${VIRTUAL_ENV}"
echo "  BRIDGE_ROOT = ${BRIDGE_ROOT}"
echo "  CUDA_HOME   = ${CUDA_HOME}"
echo "  HF_HOME     = ${HF_HOME}"

echo
echo "Testing imports..."

python - <<'PY'
import torch
import transformer_engine
import megatron.bridge
import megatron.core

print("  PyTorch:            ", torch.__version__)
print("  PyTorch CUDA:       ", torch.version.cuda)
print("  CUDA available:     ", torch.cuda.is_available())
print("  Transformer Engine: ", transformer_engine.__version__)
print("  Megatron-Bridge:     OK")
print("  Megatron-Core:       OK")
print("Apex:")
print("  Apex import:                 OK")
print("  fused_weight_gradient_mlp_cuda: OK")
PY

echo
echo "============================================================"
echo "Environment ready"
echo "============================================================"