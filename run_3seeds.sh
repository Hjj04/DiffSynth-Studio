#!/usr/bin/env bash
set -e

# --- Configuration ---
MODEL_ID="Wan-AI/Wan2.1-T2V-1.3B"
MAX_STEPS=5000  # A medium-length run
BATCH_SIZE=1
LEARNING_RATE=1e-4
BASE_LOG_DIR="./runs/multi_seed_experiments"

METADATA_CSV="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/final_inkwash_dataset/metadata.csv"
VIDEOS_ROOT_DIR="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/final_inkwash_dataset/videos"
LORA_PATH="/share/project/chengweiwu/code/Chinese_ink/hanzhe/ink_wash/lora_outputs/inkwash_style_v1/epoch-18.safetensors"
NUM_WORKERS=4

# --- Seeds to run ---
SEEDS=(111 222 333)

echo "Starting multi-seed training runs..."

for SEED in "${SEEDS[@]}"; do
  LOGDIR="${BASE_LOG_DIR}/seed_${SEED}"
  
  echo "---------------------------------"
  echo "Launching run for SEED=${SEED}"
  echo "Log directory: ${LOGDIR}"
  echo "---------------------------------"
  
  # Run sequentially. Remove the '#' from 'wait' and add '&' at the end of python command for parallel execution.
  python3 examples/wanvideo/model_training/train_with_temporal.py \
    --use_pipe \
    --model_id "${MODEL_ID}" \
    --use_flow_predictor \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --seed ${SEED} \
    --logdir ${LOGDIR} \
    --metadata_csv_path "${METADATA_CSV}" \
    --videos_root_dir "${VIDEOS_ROOT_DIR}" \
    --num_workers ${NUM_WORKERS} \
    --train_lora \
    --lora_path "${LORA_PATH}"
    
  # To run in parallel on multiple GPUs, you would typically use '&' and manage CUDA_VISIBLE_DEVICES.
  # For a single GPU, sequential execution is required.
done

# wait # Uncomment if running in parallel
echo "All multi-seed runs are complete."
