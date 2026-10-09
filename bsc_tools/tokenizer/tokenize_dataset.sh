#!/bin/bash

USER_NAME="bsc082280"
ENV_ACTIVATION_SCRIPT="/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge/bsc_tools/environments/activate_megatron_bridge.sh"
BRIDGE_ROOT="/gpfs/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge"
MODEL_ID="Qwen/Qwen3-8B"
JSONL_DATA="/home/bsc/${USER_NAME}/data/roneneldan_TinyStories/original_data/train.jsonl"
OUTPUT_FOLDER="/home/bsc/${USER_NAME}/data/roneneldan_TinyStories/tokenized_data/${MODEL_ID}/tinystories"

cd "${BRIDGE_ROOT}"

source "${ENV_ACTIVATION_SCRIPT}"

cd "${BRIDGE_ROOT}/3rdparty/Megatron-LM"

python tools/preprocess_data.py \
    --input "${JSONL_DATA}" \
    --output-prefix "${OUTPUT_FOLDER}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${MODEL_ID}" \
    --json-keys text \
    --workers 8

cd "${BRIDGE_ROOT}"