#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
################################################################################################
##################################### TUNEABLE ARGS ############################################
TARGET='*.q_proj,*.k_proj,*.v_proj,*.mm_projector'
QUOTA=30 # in precentage
STRATEGY=global_highest # [balanced, global_highest]
NUM_TRAIN_EPOCHS=1
MODEL_PATH=/mnt/models/liuhaotian/llava-v1.5-7b
OUTPUT_ROOT=outputs/aware/LLaVA-7B-AWARe-ct
CONFIG_NAME="mmmu_mllm_dcl_$(echo "$TARGET" | sed 's/\*\.//g; s/\([^,]\)[^,]*,*/\1/g')"
OUTPUT_DIR="${OUTPUT_ROOT}/${CONFIG_NAME}"
TASKS=(RS Med AD Sci Fin)
################################################################################################


################################################################################################
##################################### ANALYSIS STEP ############################################
ANALYSIS_DATASETS="$OUTPUT_DIR/analysis_ds/mmmu_sample.jsonl"
ANALYSIS_FILE=analysis.pt

train_task() {
    local dataset_name="$1"
    local base_model_path="$2"
    local target="${3:-$TARGET}"
    local quota="${4:-$QUOTA}"
    local strategy="${5:-$STRATEGY}"
    local num_train_epochs="${6:-$NUM_TRAIN_EPOCHS}"

    local select_out_name="${strategy}_${quota}"
    local task_work_dir="${OUTPUT_DIR}/${dataset_name}"
    local run_name="mllm-dcl-${dataset_name}-${strategy}-${quota}-e${num_train_epochs}"
    local model_save_path="${task_work_dir}/llava-${run_name}"
    local pos_path="${task_work_dir}/${select_out_name}.json"
    local restored_model_path="${model_save_path}-restored"

    mkdir -p "$task_work_dir"

    echo "[${dataset_name}] ===================================" >&2
    echo "[${dataset_name}] base model path: ${base_model_path}" >&2
    echo "[${dataset_name}] analysis dir: ${task_work_dir}" >&2
    echo "[${dataset_name}] run name: ${run_name}" >&2
    echo "[${dataset_name}] analysis file: ${ANALYSIS_DATASETS}" >&2
    
    accelerate launch AWARe/analyse/analyse_activation.py \
        --model-path "$base_model_path" \
        --output-dir "$task_work_dir" \
        --batch-size 1 \
        --parallel-mode data \
        --use-flash-attention \
        --question-file "$ANALYSIS_DATASETS" \
        --target "$target" >&2

    if [ $? -ne 0 ]; then
        echo "Analyse step failed for task ${dataset_name}. Exiting." >&2
        exit 1
    fi

    bash scripts/AWARe/select.sh "$OUTPUT_DIR" "$dataset_name" \
        "$ANALYSIS_FILE" "$quota" "$strategy" "$select_out_name" >&2

    if [ $? -ne 0 ]; then
        echo "Select step failed for task ${dataset_name}. Exiting." >&2
        exit 1
    fi

    local strategy_short
    strategy_short=$(echo "$strategy" | sed -E 's/([a-z])[a-z]*/\1/g; s/_//g')
    export WANDB_NAME="AWARe-${dataset_name}-${strategy_short}${quota}-e${num_train_epochs}"
    echo "[${dataset_name}] wandb name: ${WANDB_NAME}" >&2
    echo "[${dataset_name}] model save path: ${model_save_path}" >&2

    bash scripts/AWARe/train.sh "$dataset_name" "$model_save_path" \
        "$pos_path" "$num_train_epochs" "$base_model_path" >&2

    if [ $? -ne 0 ]; then
        echo "Training step failed for task ${dataset_name}. Exiting." >&2
        exit 1
    fi

    bash scripts/AWARe/restore.sh "$model_save_path" "$pos_path" >&2

    if [ $? -ne 0 ]; then
        echo "Restore step failed for task ${dataset_name}. Exiting." >&2
        exit 1
    fi

    printf '%s\n' "$restored_model_path"
}

# Prepare analysis dataset for MMMU
mkdir -p "$OUTPUT_DIR/analysis_ds"
python AWARe/analyse/prepare_mmmu.py \
    --output_dir "$OUTPUT_DIR/analysis_ds" \
    --num_samples_per_subtask 5

# Prepare analysis dataset for MLLM-DCL
python AWARe/analyse/prepare_analyse_ds_mllm_dcl.py \
    --output_dir "$OUTPUT_DIR/analysis_ds" \
    --each_ds_num 100

CURRENT_MODEL_PATH="$MODEL_PATH"
for task_name in "${TASKS[@]}"; do
    echo "============================================================" >&2
    echo "Starting continual fine-tuning task: $task_name" >&2
    echo "CURRENT_MODEL_PATH: $CURRENT_MODEL_PATH" >&2
    CURRENT_MODEL_PATH=$(train_task "$task_name" "$CURRENT_MODEL_PATH")
    if [ $? -ne 0 ]; then
        echo "Task $task_name failed. Exiting." >&2
        exit 1
    fi
    ANALYSIS_DATASETS+=",${OUTPUT_DIR}/analysis_ds/${task_name}.jsonl"
done

echo "All tasks finished. Final restored model: $CURRENT_MODEL_PATH" >&2