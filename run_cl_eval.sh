#!/bin/bash

TASKS=(RS Med AD Sci Fin)

OUTPUT_DIR="outputs/aware/LLaVA-7B-AWARe-ct/mmmu_cl_qkvm"

for task_name in "${TASKS[@]}"; do
    echo "============================================================" >&2
    echo "Starting eval $task_name model" >&2
    bash scripts/AWARe/eval_cl.sh "$OUTPUT_DIR/$task_name/llava-cl-$task_name-global_highest-30-e1-restored" "results/cl" "$task_name"
done
