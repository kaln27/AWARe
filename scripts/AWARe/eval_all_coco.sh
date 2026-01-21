#!/bin/bash

MODEL_RESTORE_PATH=${1:-''}

##########################################################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Ues relative path here
MODEL_PATH_ABS=`realpath $MODEL_RESTORE_PATH`
RESULT_DIR=${MODEL_PATH_ABS}/eval_results
SUMMARY_FILE=${RESULT_DIR}/summary.txt
echo "Eval all coco args:"
echo "  MODEL_PATH: $MODEL_RESTORE_PATH"
##########################################################################################


##########################################################################################
##### Summary file #####
mkdir -p $RESULT_DIR
current_time=$(date '+%Y-%m-%d %H:%M:%S')
> "$SUMMARY_FILE"    # Clear out the output file if it exists.
echo "Current Time: $current_time" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
##########################################################################################

EVAL_ON_COCO=scripts/AWARe/downstream/eval_coco.sh

EVAL_ON_GQA=scripts/AWARe/upstream/eval_gqa.sh
EVAL_ON_OKVQA=scripts/AWARe/upstream/eval_okvqa.sh
EVAL_ON_OCRVQA=scripts/AWARe/upstream/eval_ocrvqa.sh
EVAL_ON_TEXTVQA=scripts/AWARe/upstream/eval_textvqa.sh


bash $EVAL_ON_COCO $MODEL_PATH_ABS $RESULT_DIR $SUMMARY_FILE

bash $EVAL_ON_OKVQA $MODEL_PATH_ABS $RESULT_DIR $SUMMARY_FILE
bash $EVAL_ON_OCRVQA $MODEL_PATH_ABS $RESULT_DIR $SUMMARY_FILE
bash $EVAL_ON_GQA $MODEL_PATH_ABS $RESULT_DIR $SUMMARY_FILE
bash $EVAL_ON_TEXTVQA $MODEL_PATH_ABS $RESULT_DIR $SUMMARY_FILE