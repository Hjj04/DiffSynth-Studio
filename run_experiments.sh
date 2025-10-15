#!/bin/bash

# =================================================================
#  Automated Experiment and Evaluation Runner
# =================================================================
# This script runs a series of predefined experiments, evaluates their
# outputs, and compiles the results into a single summary CSV file.

# --- Configuration ---
# All command-line arguments passed to this script will be added to every experiment.
# Example: ./run_experiments.sh --max_steps 1000 --batch_size 2
EXTRA_FLAGS="$@"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_LOG_DIR="./runs/auto_experiments_${TIMESTAMP}"
SUMMARY_FILE="${BASE_LOG_DIR}/summary_report.csv"
mkdir -p "$BASE_LOG_DIR"

# A function to run a single experiment
run_experiment() {
    local name="$1"
    local flags="$2"
    local logdir="${BASE_LOG_DIR}/${name}"
    
    echo "========================"
    echo "Running experiment: ${name}"
    echo "Log directory: ${logdir}"
    echo "Extra flags: ${flags} ${EXTRA_FLAGS}"
    echo "------------------------"
    
    # Construct and run the training command
    CMD="python3 examples/wanvideo/model_training/train_with_temporal.py \
        --logdir ${logdir} \
        ${flags} \
        ${EXTRA_FLAGS}"
    
    echo "Executing: ${CMD}"
    eval ${CMD}
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        echo "Training FAILED for experiment: ${name}"
        return
    fi
    
    echo "Training finished for ${name}. Running evaluation..."
    
    # Run the evaluation script
    python3 examples/wanvideo/model_training/evaluate_metrics.py "${logdir}"
    
    # Append results to the summary file
    local metrics_csv="${logdir}/metrics.csv"
    if [ -f "$metrics_csv" ]; then
        if [ ! -f "$SUMMARY_FILE" ]; then
            # Add header to summary file if it doesn't exist
            echo "experiment_name,$(cat $metrics_csv)" > "$SUMMARY_FILE"
        fi
        # Add the experiment name and its metrics to the summary
        echo "${name},$(tail -n 1 $metrics_csv)" >> "$SUMMARY_FILE"
    fi
}

# --- Define and Run Experiments ---

# Experiment 1: Baseline using the pipeline, but no temporal module enhancements (flow=None).
run_experiment "baseline_pipe_no_flow" \
    "--use_pipe"

# Experiment 2: Pipeline with the learned latent flow predictor.
run_experiment "pipe_with_learned_flow" \
    "--use_pipe --use_flow_predictor"

# Experiment 3: Training only LoRA with the pipeline and flow predictor.
# NOTE: Update the lora_path to your actual file location.
LORA_PATH="/share/project/chengweiwu/code/Chinese_ink/hanzhe/models/inkwash_style_lora_v1/epoch-99.safensors"
if [ -f "$LORA_PATH" ]; then
    run_experiment "lora_pipe_with_learned_flow" \
        "--use_pipe --use_flow_predictor --train_lora --lora_path ${LORA_PATH}"
else
    echo "Warning: LoRA file not found at ${LORA_PATH}. Skipping LoRA experiment."
fi


echo "========================"
echo "All experiments finished."
echo "Summary report available at: ${SUMMARY_FILE}"
echo "========================"
cat "${SUMMARY_FILE}" | csvlook