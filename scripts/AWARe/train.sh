#! /usr/bin/bash

DATASET_NAME=${1:-'iconqa'}  # iconqa or coco

DEVICE="localhost:${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset CUDA_VISIBLE_DEVICES
DEEPSPEED_ZEROFILE=./scripts/zero2.json
VISION_TOWER_PATH=/mnt/models/openai/clip-vit-large-patch14-336
MODEL_SAVE_PATH=${2:-''}
POS_PATH=${3:-''}
NUM_TRAIN_EPOCHS=${4:-3}
MODEL_NAME_OR_PATH=${5:-"/mnt/models/liuhaotian/llava-v1.5-7b"}

echo "Training args:"
echo "  DEVICE: $DEVICE"
echo "  DATASET_NAME: $DATASET_NAME"
echo "  MODEL_SAVE_PATH: $MODEL_SAVE_PATH"
echo "  POS_PATH: $POS_PATH"
echo "  NUM_TRAIN_EPOCHS: $NUM_TRAIN_EPOCHS"

# Make sure total batch size = 16 [Device num * per device batch size * gradient accumulation steps]
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=2e-5
MM_PROJ_LR=2e-4
SAVE_STEPS=100000
SAVE_TOTAL_LIMIT=10

if [ "$DATASET_NAME" == "iconqa" ]; then
    data_path="instructions/IconQA_txt/iconqa_txt-train.json"
    image_folder="datasets/iconqa_data"
elif [ "$DATASET_NAME" == "coco" ]; then
    data_path="instructions/COCO-Caption/coco-train.json"
    image_folder="datasets/COCO/train2014"
elif [ "$DATASET_NAME" == "RS" ]; then
    data_path="datasets/RS/train.json"
    image_folder="datasets/RS"
elif [ "$DATASET_NAME" == "Med" ]; then
    data_path="datasets/Med/train.json"
    image_folder="datasets/Med"
elif [ "$DATASET_NAME" == "AD" ]; then
    data_path="datasets/AD/train.json"
    image_folder="datasets/AD"
elif [ "$DATASET_NAME" == "Sci" ]; then
    data_path="datasets/Sci/train.json"
    image_folder="datasets/Sci"
elif [ "$DATASET_NAME" == "Fin" ]; then
    data_path="datasets/Fin/train.json"
    image_folder="datasets/Fin"
elif [ "$DATASET_NAME" == "ImageNet-R" ]; then
    data_path="instructions/ImageNet-R/train.json"
    image_folder="datasets"
elif [ "$DATASET_NAME" == "ArxivQA" ]; then
    data_path="instructions/ArxivQA/train_4w.json"
    image_folder="datasets"
elif [ "$DATASET_NAME" == "VizWiz" ]; then
    data_path="instructions/VizWiz/train.json"
    image_folder="datasets"
elif [ "$DATASET_NAME" == "IconQA" ]; then
    data_path="instructions/IconQA/train.json"
    image_folder="datasets"
elif [ "$DATASET_NAME" == "CLEVR-Math" ]; then
    data_path="instructions/CLEVR-Math/train_4w.json"
    image_folder="datasets"
elif [ "$DATASET_NAME" == "Flickr30k" ]; then
    data_path="instructions/Flickr30k/train_brief_4w.json"
    image_folder="datasets"
else
    echo "Unsupported DATASET_NAME: $DATASET_NAME"
    exit 1
fi

deepspeed --include $DEVICE llava/train/train_mem.py \
    --deepspeed $DEEPSPEED_ZEROFILE \
    --use_aware True \
    --pos_path $POS_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower $VISION_TOWER_PATH \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $MODEL_SAVE_PATH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LEARNING_RATE \
    --mm_projector_lr $MM_PROJ_LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb