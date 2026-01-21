#! /usr/bin/bash

MODEL_PATH=/mnt/models/liuhaotian/llava-v1.5-7b
OUTPUT_DIR=${1:-'outputs/aware/LLaVA-7B-AWARe'}
SUBDIR=${2:-'200_200_200_200_qkvm'}
DS_OUTPUT_PATH=$OUTPUT_DIR/$SUBDIR/${3:-'analyse_ds.jsonl'}
TARGET_DS=${4:-'okvqa,ocrvqa,gqa,textvqa'}
EACH_DS_NUM=${5:-'200,200,200,200'}
TARGET=${6:-'*.q_proj,*.k_proj,*.v_proj,*.mm_projector'}

mkdir -p $OUTPUT_DIR/$SUBDIR

echo "Analyse args:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  SUBDIR: $SUBDIR"
echo "  DS_OUTPUT_PATH: $DS_OUTPUT_PATH"
echo "  TARGET_DS: $TARGET_DS"
echo "  EACH_DS_NUM: $EACH_DS_NUM"
echo "  TARGET: $TARGET"

# sub_dir name explanation:
# trainable_weight: q (q_proj), k (k_proj), v (v_proj), o (o_proj),
#                   u (up_proj), g (gate_proj), d (down_proj) m (mm_projector)
# analyse_dataset: okvqa, ocrvqa, gqa, textvqa
# each_ds_num: number of samples per dataset, split by comma or int

# target_ds: 'okvqa,ocrvqa,gqa,textvqa'
# prepare data for AWARe analysis
python AWARe/analyse/prepare_analyse_ds.py \
    --output_path $DS_OUTPUT_PATH \
    --target_ds $TARGET_DS \
    --each_ds_num $EACH_DS_NUM

python AWARe/analyse/analyse_activation.py \
    --model-path $MODEL_PATH \
    --output-dir $OUTPUT_DIR/$SUBDIR \
    --batch-size 1 \
    --use-flash-attention \
    --question-file $DS_OUTPUT_PATH \
    --target $TARGET