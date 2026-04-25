#!/bin/bash

MODEL_PATH=${1:-""}
RESULT_DIR=${2:-""}
SUMMARY_FILE=${3:-"None"}
GPU_NUM=${4:-4}

QUESTION_FILE="datasets/AD/test.json"
IMAGE_FOLDER="datasets/AD"
ANNOTATION_FILE="$QUESTION_FILE"

echo "Eval AD args:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  RESULT_DIR: $RESULT_DIR"
echo "  SUMMARY_FILE: $SUMMARY_FILE"
echo "  GPU_NUM: $GPU_NUM"
echo "  STAGE: $STAGE"

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

mkdir -p "$RESULT_DIR/AD"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_ai2d \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$RESULT_DIR/AD/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file="$RESULT_DIR/AD/merge.jsonl"
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$RESULT_DIR/AD/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python -m llava.eval.CoIN.eval_ai2d \
    --annotation-file "$ANNOTATION_FILE" \
    --result-file "$output_file" \
    --summary-file "$SUMMARY_FILE"
