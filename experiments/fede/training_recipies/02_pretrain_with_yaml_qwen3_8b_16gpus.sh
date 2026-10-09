#!/bin/bash

###############################################################################
# Qwen3 8B pre-training — 16 GPUs / 4 nodes
#
# This script:
#   1. Requests 4 MN5 nodes with 4 GPUs each.
#   2. Activates the Megatron-Bridge environment.
#   3. Launches the training recipe with torchrun.
#
# The training configuration is defined in the YAML file referenced below.
###############################################################################

###############################################################################
# SLURM configuration
###############################################################################

# IMPORTANT: You should change these variables to adapt the script to your needs

#SBATCH -D .
#SBATCH -A bsc88
#SBATCH -q acc_debug
############# Obligatorias #######################
#SBATCH --time=0-02:00:00               # Consultar batchlim para entender los límites de las particiones. 
################# HOST ###########################
#SBATCH --nodes=4                       # Número de nodos
#SBATCH --ntasks=4                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=160              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:4
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=none
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
################ Logging #########################
#SBATCH --job-name=qwen3_8b_16gpus
#SBATCH --verbose
#SBATCH --output=/gpfs/scratch/bsc88/bsc082280/outputs/logs/%x/%j.txt
################ Job notifications #########################
#SBATCH --mail-type=none
#SBATCH --mail-user=federico.costa@bsc.es

###############################################################################

###############################################################################
# User / repository configuration
###############################################################################

# IMPORTANT: You should change these variables to adapt the script to your needs

USER_NAME="bsc082280"
JOB_NAME="qwen3_8b_16gpus"
ENV_ACTIVATION_SCRIPT="/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge/bsc_tools/environments/activate_megatron_bridge.sh"
BRIDGE_ROOT="/gpfs/home/bsc/${USER_NAME}/git_repositories/BSC-Megatron-Bridge"
TRAINING_RECIPE_PATH="${BRIDGE_ROOT}/experiments/fede/training_recipies/02_pretrain_with_yaml_qwen3_8b_16gpus.py"
CONFIG_FILE_PATH="${BRIDGE_ROOT}/experiments/fede/experiments_configs/qwen3_8b_pretrain_16gpus.yaml"

OUTPUT_LOG_PATH="/home/bsc/${USER_NAME}/outputs/logs/$(date +%Y-%m-%d_%H-%M-%S)_${JOB_NAME}.txt"

###############################################################################

###############################################################################
# Distributed training configuration
###############################################################################

# These are fixed for this job, you shouldn't change these
NNODES=4
NTASKS=4
NTASKS_PER_NODE=1
NPROC_PER_NODE=4
EXPERIMENT_DATETIME=$(date +"%Y_%m_%d_%H_%M_%S")

###############################################################################

echo "Start datetime: $(date +%Y-%m-%d_%H:%M:%S)"

export EXPERIMENT_DATETIME
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) | grep "Address: " | sed "s/Address: //" | tail)
export MASTER_PORT=20074
# When using srun, the SLURM_CPUS_PER_TASK variable is no longer taken into consideration, as it also doesn't --cpus-per-task
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export NPROC_PER_NODE
export NNODES
export TRAINING_RECIPE_PATH
export CONFIG_FILE_PATH

source "${ENV_ACTIVATION_SCRIPT}"
cd "${BRIDGE_ROOT}"

echo "============================================================"
echo "Job information"
echo "============================================================"
echo "Start datetime:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Job name:        ${SLURM_JOB_NAME}"
echo "Node list:       ${SLURM_JOB_NODELIST}"
echo "Nodes:           ${SLURM_NNODES}"
echo "CPUs per task:   ${SLURM_CPUS_PER_TASK}"
echo "GPUs requested:  ${NPROC_PER_NODE}"
echo "Bridge root:       ${BRIDGE_ROOT}"
echo "Training recipe:   ${TRAINING_RECIPE_PATH}"
echo "Config file:       ${CONFIG_FILE_PATH}"
echo "Master address:    ${MASTER_ADDR}"
echo "Master port:       ${MASTER_PORT}"
echo "Output log:        ${OUTPUT_LOG_PATH}"
echo "============================================================"

srun \
    --ntasks=${NTASKS} \
    --ntasks-per-node=${NTASKS_PER_NODE} \
    bash -c '
        echo "HOST=$(hostname) SLURM_NODEID=$SLURM_NODEID"

        torchrun \
            --nproc_per_node $NPROC_PER_NODE \
            --nnodes $NNODES \
            --node-rank $SLURM_NODEID \
            --master-addr $MASTER_ADDR \
            --master-port $MASTER_PORT \
            --max-restarts 0 \
            $TRAINING_RECIPE_PATH \
            --config-file $CONFIG_FILE_PATH
    ' 2>&1 | tee ${OUTPUT_LOG_PATH}

echo "============================================================"
echo "Training finished"
echo "End datetime: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"