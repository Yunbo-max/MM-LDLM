#!/bin/bash
# File: train_l2t.sh
set -xeuo pipefail

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Try to get platform environment variables with defaults
NODE_RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Reduce allocator fragmentation and avoid wandb login prompts / unwritable paths
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_DIR="./output_dir/wandb"
mkdir -p "${WANDB_DIR}"

# Relax NCCL watchdog to mitigate false-positive hangs
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Training Latent-to-Text (l2t) model"
echo "Node Rank: ${NODE_RANK}"
echo "World Size: ${WORLD_SIZE}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"

# Create output directory
mkdir -p output_dir/l2t_logs

# Run training with l2t mode - Using your corrected command
if [ ${WORLD_SIZE} -gt 1 ]; then
    # Multi-node
    torchrun \
      --nnodes=${WORLD_SIZE} \
      --nproc_per_node=2 \
      --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      latentDLM_mmdit/train_mmdit.py \
      --config-name mmdit_preprocessed \
      logging.run_name="mmdit-1.2-newloader-fixed" \
      logging.save_dir="./output_dir/l2t_models" \
      training.train_batch_size=16 \
      training.eval_batch_size=16 \
      training.num_train_steps=1000000 \
      training.compile_model=true \
      training.loss_type="l2t" \
      model.latent_dim=1024 \
      training.dtype=bf16 \
      logging.save_freq=50000 \
      logging.log_freq=10000 \
      logging.eval_freq=10000 \
      data.num_workers=8 \
      optimizer.lr=1e-4
else
    # Single-node - Using your corrected command parameters
    torchrun \
      --nnodes=1 \
      --nproc_per_node=2 \
      latentDLM_mmdit/train_mmdit.py \
      --config-name mmdit_preprocessed \
      logging.run_name="mmdit-1.2-newloader-fixed" \
      logging.save_dir="./output_dir/l2t_models" \
      training.train_batch_size=16 \
      training.eval_batch_size=16 \
      training.num_train_steps=1000000 \
      training.compile_model=true \
      training.loss_type="l2t" \
      model.latent_dim=1024 \
      training.dtype=bf16 \
      logging.save_freq=50000 \
      logging.log_freq=10000 \
      logging.eval_freq=10000 \
      data.num_workers=8 \
      optimizer.lr=1e-4
fi

# Wait for l2t training to complete
echo "=========================================="
echo "L2T training completed!"
echo "Waiting 10 seconds before starting T2L..."
echo "=========================================="
sleep 10

# ============== T2L Training ==============
echo "=========================================="
echo "2. Training Text-to-Latent (t2l) model"
echo "=========================================="
echo "Node Rank: ${NODE_RANK}"
echo "World Size: ${WORLD_SIZE}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: $((MASTER_PORT + 1))"  # Use different port
T2L_PORT=$((MASTER_PORT + 1))

# Create output directory
mkdir -p output_dir/t2l_logs

# Run training with t2l mode
if [ ${WORLD_SIZE} -gt 1 ]; then
    # Multi-node - Keeping t2l with original parameters for now
    torchrun \
      --nnodes=${WORLD_SIZE} \
      --nproc_per_node=2 \
      --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${T2L_PORT} \
      latentDLM_mmdit/train_mmdit.py \
      --config-name mmdit \
      logging.run_name="mmdit-t2l-training" \
      logging.save_dir="./output_dir/t2l_models" \
      training.train_batch_size=32 \
      training.eval_batch_size=32 \
      training.num_train_steps=1131000 \
      training.compile_model=false \
      training.loss_type="t2l" \
      model.latent_dim=1024 \
      training.dtype=bf16 \
      logging.save_freq=5000 \
      logging.log_freq=100 \
      logging.eval_freq=1000 \
      ptimizer.lr=1e-5
else
    # Single-node - Keeping t2l with original parameters for now
    torchrun \
      --nproc_per_node=2 \
      latentDLM_mmdit/train_mmdit.py \
      --config-name mmdit \
      logging.run_name="mmdit-t2l-training" \
      logging.save_dir="./output_dir/t2l_models" \
      training.train_batch_size=32 \
      training.eval_batch_size=32 \
      training.num_train_steps=1131000 \
      training.compile_model=false \
      training.loss_type="t2l" \
      model.latent_dim=1024 \
      training.dtype=bf16 \
      logging.save_freq=5000 \
      logging.log_freq=100 \
      logging.eval_freq=1000 \
      ptimizer.lr=1e-5
fi