################################################################################################
##################################### TUNEABLE ARGS ############################################
ANALYSIS=a
TARGET='*.q_proj,*.k_proj,*.v_proj,*.mm_projector'
QUOTA=30 # in precentage
STRATEGY=global_highest # [balanced, global_highest]
DATASET_NAME=iconqa # iconqa or coco
NUM_TRAIN_EPOCHS=3
MODEL_PATH=liuhaotian/llava-v1.5-7b
################################################################################################

# OUTPUT_DIR name explanation:
# LLaVA-7B-AWARe-xwya
# xwya means analysis = 0.x * weight + 0.y * activation  

# SUBDIR name explanation:
# trainable_weight: q (q_proj), k (k_proj), v (v_proj), o (o_proj),
#                   u (up_proj), g (gate_proj), d (down_proj) m (mm_projector)

################################################################################################
##################################### ANALYSIS STEP ############################################
OUTPUT_DIR=outputs/aware/LLaVA-7B-AWARe-${ANALYSIS}
SUBDIR=''
DS_OUTPUT_PATH=outputs/mmmu_sample/mmmu_sample.jsonl
TARGET=$TARGET

SUBDIR="mmmu_$(echo $TARGET | sed 's/\*\.//g; s/\([^,]\)[^,]*,*/\1/g')"

python AWARe/analyse/prepare_mmmu.py
python AWARe/analyse/analyse_activation.py \
    --model-path $MODEL_PATH \
    --output-dir $OUTPUT_DIR/$SUBDIR \
    --batch-size 1 \
    --use-flash-attention \
    --question-file $DS_OUTPUT_PATH \
    --target $TARGET
################################################################################################

if [ $? -ne 0 ]; then
    echo "Analyse step failed. Exiting."
    exit 1
fi

################################################################################################
####################################### SELECT STEP ############################################
OUTPUT_DIR=$OUTPUT_DIR
SUBDIR=$SUBDIR
ANALYSIS_FILE=analysis.pt
QUOTA=$QUOTA # in precentage
STRATEGY=$STRATEGY # [balanced, global_highest]
SELECT_OUT_NAME="${STRATEGY}_${QUOTA}"

bash scripts/AWARe/select.sh $OUTPUT_DIR $SUBDIR $ANALYSIS_FILE $QUOTA $STRATEGY $SELECT_OUT_NAME
################################################################################################

if [ $? -ne 0 ]; then
    echo "Select step failed. Exiting."
    exit 1
fi

################################################################################################
######################################## TRAIN STEP ############################################
DATASET_NAME=$DATASET_NAME # iconqa or coco
MODEL_SAVE_PATH=$OUTPUT_DIR/$SUBDIR/llava-${SELECT_OUT_NAME}-e${NUM_TRAIN_EPOCHS}-${DATASET_NAME}
POS_PATH=$OUTPUT_DIR/$SUBDIR/$SELECT_OUT_NAME.json

STRATEGY_SHORT=$(echo $STRATEGY | sed -E 's/([a-z])[a-z]*/\1/g; s/_//g')
export WANDB_NAME="AWARe-${ANALYSIS}-${SUBDIR//_/-}-${STRATEGY_SHORT}${QUOTA}-e${NUM_TRAIN_EPOCHS}-${DATASET_NAME}"
echo WANDB_NAME: $WANDB_NAME
    
# bash scripts/AWARe/train.sh $DATASET_NAME $MODEL_SAVE_PATH $POS_PATH $NUM_TRAIN_EPOCHS
################################################################################################

if [ $? -ne 0 ]; then
    echo "Training step failed. Exiting."
    exit 1
fi

################################################################################################
####################################### RESTORE STEP ###########################################
MODEL_SAVE_PATH=$MODEL_SAVE_PATH
POS_PATH=$POS_PATH

bash scripts/AWARe/restore.sh $MODEL_SAVE_PATH $POS_PATH
################################################################################################

if [ $? -ne 0 ]; then
    echo "Restore step failed. Exiting."
    exit 1
fi

################################################################################################
######################################## EVAL STEP #############################################
MODEL_RESTORE_PATH=${MODEL_SAVE_PATH}-restored

if [ "$DATASET_NAME" = "iconqa" ]; then
    bash scripts/AWARe/eval_all_iconqa.sh $MODEL_RESTORE_PATH
elif [ "$DATASET_NAME" = "coco" ]; then
    bash scripts/AWARe/eval_all_coco.sh $MODEL_RESTORE_PATH
else
    echo "Unsupported DATASET_NAME: $DATASET_NAME"
fi
################################################################################################