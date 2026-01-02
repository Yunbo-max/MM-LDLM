#!/bin/bash
set -xeuo pipefail

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sonar

# Try to get platform environment variables with defaults
NODE_RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-2}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Node Rank: ${NODE_RANK}"
echo "World Size: ${WORLD_SIZE}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"

# Create output directory
mkdir -p preprocessed_data/sonar_1024d_test

# Run data preprocessing
if [ ${WORLD_SIZE} -gt 1 ]; then
    # Multi-node data preprocessing
    echo "Running multi-node data preprocessing with ${WORLD_SIZE} nodes..."
    torchrun \
      --nnodes=${WORLD_SIZE} \
      --nproc_per_node=4 \
      --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      preprocessed_data/prepare_data_sonar.py \
      --datasets openwebtext \
      --latent-model sonar \
      --batch-size 1024 \
      --max-samples 1000000000 \
      --output-dir preprocessed_data/sonar_1024d_test
else
    # Single-node data preprocessing
    echo "Running single-node data preprocessing..."
    torchrun \
      --nnodes=1 \
      --nproc_per_node=2 \
      preprocessed_data/prepare_data_sonar.py \
      --datasets openwebtext \
      --latent-model sonar \
      --batch-size 1024 \
      --max-samples 1000000000 \
      --output-dir preprocessed_data/sonar_1024d_test
fi