#!/bin/bash

USER_NAME="bsc082280"
ENV_ACTIVATION_SCRIPT="/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge/bsc_tools/environments/activate_megatron_bridge.sh"
BRIDGE_ROOT="/gpfs/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge"

export TRANSFORMERS_OFFLINE=0

# Set your Hugging Face Hub token here if needed
#export HUGGING_FACE_HUB_TOKEN=xxx

cd "${BRIDGE_ROOT}"

python bsc_tools/utils/download_from_hf.py


