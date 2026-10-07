#!/bin/bash
#SBATCH -D .
#SBATCH -A bsc88
#SBATCH -q acc_debug
############# Obligatorias #######################
#SBATCH --time=0-02:00:00               # Consultar batchlim para entender los límites de las particiones. 
################# HOST ###########################
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=160              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:4
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=none
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
################ Logging #########################
#SBATCH --job-name=qwen3_8b_4gpus
#SBATCH --verbose
#SBATCH --output=/gpfs/scratch/bsc88/bsc082280/outputs/logs/%x/%j.txt
################ Job notifications #########################
#SBATCH --mail-type=none
#SBATCH --mail-user=federico.costa@bsc.es

echo "Start datetime: $(date +%Y-%m-%d_%H:%M:%S)"

EXPERIMENT_DATETIME=$(date +"%Y_%m_%d_%H_%M_%S")

export EXPERIMENT_DATETIME

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) | grep "Address: " | sed "s/Address: //" | tail)
export MASTER_PORT=20074
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

cd /home/bsc/bsc082280/git_repositories/BSC-Megatron-Bridge
source /bsc_tools/environments/activate_megatron_bridge.sh

srun \
    --ntasks=1 \
    --ntasks-per-node=1 \
    bash -c '
        echo "HOST=$(hostname) SLURM_NODEID=$SLURM_NODEID"

        torchrun \
            --nproc_per_node=4 \
            --nnodes=1 \
            --node-rank $SLURM_NODEID \
            --master-addr $MASTER_ADDR \
            --master-port $MASTER_PORT \
            --max-restarts 0 \
            /experiments/fede/training_recipies/02_pretrain_with_yaml_qwen3_8b_4gpus.py \
            --config-file /experiments/fede/experiments_configs/qwen3_8b_pretrain_4gpus.yaml
    ' 2>&1 | tee /home/bsc/bsc082280/outputs/logs/$(date +%Y-%m-%d_%H-%M-%S)_02_pretrain_with_yaml.txt

echo "End datetime: $(date +%Y-%m-%d_%H:%M:%S)"