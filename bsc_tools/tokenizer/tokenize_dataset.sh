#!/bin/bash

cd /home/bsc/bsc082280/git_repositories/BSC-Megatron-Bridge

# Environment
source /home/bsc/bsc082280/git_repositories/BSC-Megatron-Bridge/bsc_tools/environments/activate_megatron_bridge.sh

cd /home/bsc/bsc082280/git_repositories/BSC-Megatron-Bridge/3rdparty/Megatron-LM

python tools/preprocess_data.py \
    --input /home/bsc/bsc082280/data/roneneldan_TinyStories/original_data/train.jsonl \
    --output-prefix /home/bsc/bsc082280/data/roneneldan_TinyStories/tokenized_data/Qwen/Qwen3-8B/tinystories \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3-8B \
    --json-keys text \
    --workers 8

cd /home/bsc/bsc082280/git_repositories/BSC-Megatron-Bridge