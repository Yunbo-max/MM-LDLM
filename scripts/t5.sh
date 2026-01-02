#!/bin/bash
set -xeuo pipefail

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

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
mkdir -p output_dir/logs

# Run training
if [ ${WORLD_SIZE} -gt 1 ]; then
    # Multi-node
    torchrun \
      --nnodes=${WORLD_SIZE} \
      --nproc_per_node=8 \
      --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      latentDLM_mmdit/train_mmdit.py \
      --config-name mmdit \
      logging.run_name="mmdit-2node" \
      training.train_batch_size=16 \
      training.eval_batch_size=16 \
      training.num_train_steps=25000 \
      training.compile_model=false \
      model.latent_dim=1024 \
      training.dtype=bf16
else
    # Single-node
    torchrun \
      --nproc_per_node=8 \
      latentDLM_mmdit/train_mmdit.py \
      --config-name mmdit \
      logging.run_name="mmdit-single" \
      training.train_batch_size=16 \
      training.eval_batch_size=16 \
      training.num_train_steps=25000 \
      training.compile_model=false \
      model.latent_dim=1024 \
      training.dtype=bf16
fi