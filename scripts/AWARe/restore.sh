#! /usr/bin/bash

MODEL_PATH=${1:-''}
POS_PATH=${2:-''}

echo "Restore args:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  POS_PATH: $POS_PATH"

python AWARe/restore/restore_model.py \
    --model-path $MODEL_PATH \
    --model-base /mnt/models/liuhaotian/llava-v1.5-7b \
    --target-pos-path $POS_PATH \
    --output-path ${MODEL_PATH}-restored