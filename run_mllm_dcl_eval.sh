#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
TASKS=(RS Med AD Sci Fin)

OUTPUT_DIR="outputs/aware/LLaVA-7B-AWARe-ct/mmmu_mllm_dcl_qkvm"

for task_name in "${TASKS[@]}"; do
    echo "============================================================" >&2
    echo "Starting eval $task_name model" >&2
    bash scripts/AWARe/eval_mllm_dcl.sh "$OUTPUT_DIR/$task_name/llava-mllm_dcl-$task_name-global_highest-30-e1-restored" "results/mllm_dcl" "$task_name"
done
