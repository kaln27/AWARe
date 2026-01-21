#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


MODEL_PATH=""
SPLIT="ocrvqa"
RESULT_DIR=""


if [ ! -n "$1" ] ;then
    MODEL_PATH=$MODEL_PATH
else
    MODEL_PATH=$1
fi

if [ ! -n "$2" ] ;then
    RESULT_DIR=$RESULT_DIR
else
    RESULT_DIR=$2
fi

if [ ! -n "$3" ] ;then
    SUMMARY_FILE="None"
else
    SUMMARY_FILE=$3
fi

mkdir -p $RESULT_DIR

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file instructions/OCRVQA/sampled_ocrvqa_test.jsonl \
        --image-folder datasets/OCR-VQA/images \
        --answers-file $RESULT_DIR/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done


wait

output_file=$RESULT_DIR/$SPLIT/output.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python -m llava.eval.eval_ocrvqa \
    --annotation-file instructions/OCRVQA/sampled_ocrvqa_test.jsonl \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$SPLIT \
    --summary-file $SUMMARY_FILE