#!/bin/bash

# RAQUEL Hyperparameter Search Script
# Runs unlearning experiments with different methods, regularizations, and learning rates.

# ---------------- Configuration ----------------
# Hyperparameters
learning_rates=(1e-5 2e-5 3e-5 4e-5)
methods=(ga npo dpo idk)
regularizations=(gd kl)

# Model
MODEL_TAG="finetune_full-llama-1b"

# GPUs to use
gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
gpu_idx=0

# Base settings
PYTHON_CMD="python"  # Ensure this points to your environment's python
SCRIPT="script/train/unlearn.py"
# -----------------------------------------------

echo "Starting Hyperparameter Search..."
echo "Total combinations: $((${#methods[@]} * ${#regularizations[@]} * ${#learning_rates[@]}))"

# Loop through all combinations
for method in "${methods[@]}"; do
  for reg in "${regularizations[@]}"; do
    for lr in "${learning_rates[@]}"; do
      
      # Get current GPU ID from the pool
      gpu_id=${gpus[$gpu_idx]}
      
      # Create a unique tag for tracking (e.g., for Neptune/Logs)
      tag="hp_unlearn_${method}_${reg}_lr${lr}"
      
      echo "Launching: Method=${method} Reg=${reg} LR=${lr} on GPU=${gpu_id}"
      
      # Run the unlearning script in the background
      CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON_CMD} ${SCRIPT} \
        --config-name=unlearn/${method} \
        regularization=${reg} \
        training.learning_rate=${lr} \
        model.trained_tag=${MODEL_TAG} \
        tag=${tag} \
        &

      # Move to next GPU
      gpu_idx=$((gpu_idx + 1))
      
      # If we have utilized all GPUs, wait for the current batch to finish
      if [ ${gpu_idx} -ge ${num_gpus} ]; then
        echo "Batch full. Waiting for current experiments to finish..."
        wait
        echo "Batch finished. Resuming..."
        gpu_idx=0
      fi
      
    done
  done
done

# Wait for any remaining background jobs
wait
echo "All experiments completed!"
