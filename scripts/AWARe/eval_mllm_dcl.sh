#!/bin/bash

MODEL_PATH=${1:-""}
RESULT_ROOT=${2:-""}
CURRENT_TASK=${3:-"Fin"}
SUMMARY_FILE=${4:-"${RESULT_ROOT}/${CURRENT_TASK}/summary.txt"}
GPU_NUM=${5:-4}

TASK_ORDER=(RS Med AD Sci Fin)

if [ -z "$MODEL_PATH" ] || [ -z "$RESULT_ROOT" ]; then
    echo "Usage: $0 <model_path> <result_root> [current_task] [summary_file] [gpu_num]"
    exit 1
fi

CURRENT_TASK=${CURRENT_TASK^}

mkdir -p "${RESULT_ROOT}/${CURRENT_TASK}"
> "$SUMMARY_FILE"
echo "Continual learning evaluation" >> "$SUMMARY_FILE"
echo "Model path: $MODEL_PATH" >> "$SUMMARY_FILE"
echo "Current task: $CURRENT_TASK" >> "$SUMMARY_FILE"
echo "GPU num: $GPU_NUM" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

run_eval() {
    local task_name="$1"
    local eval_script="scripts/AWARe/mllm_dcl/eval_${task_name,,}.sh"

    echo "============================================================"
    echo "Evaluating task: ${task_name}"
    bash "$eval_script" "$MODEL_PATH" "$RESULT_ROOT/$CURRENT_TASK" "$SUMMARY_FILE" "$GPU_NUM"
}

for task_name in "${TASK_ORDER[@]}"; do
    run_eval "$task_name"
    if [ "$task_name" = "$CURRENT_TASK" ]; then
        break
    fi
done

echo "Evaluation finished. Summary: $SUMMARY_FILE"
