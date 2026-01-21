################################################################################################
##################################### TUNEABLE ARGS ############################################
export CUDA_VISIBLE_DEVICES=4,5,6,7
ANALYSIS=a
TARGET_DS='okvqa,ocrvqa,gqa,textvqa'
EACH_DS_NUM='200,200,200,200'
TARGET='*.q_proj,*.k_proj,*.v_proj,*.mm_projector'
QUOTA=10 # in precentage
STRATEGY=balanced # [balanced, global_highest]
DATASET_NAME=iconqa # iconqa or coco
NUM_TRAIN_EPOCHS=3
################################################################################################

# OUTPUT_DIR name explanation:
# LLaVA-7B-AWARe-xwya
# xwya means analysis = 0.x * weight + 0.y * activation  

# SUBDIR name explanation:
# trainable_weight: q (q_proj), k (k_proj), v (v_proj), o (o_proj),
#                   u (up_proj), g (gate_proj), d (down_proj) m (mm_projector)
# analyse_dataset: okvqa, ocrvqa, gqa, textvqa
# each_ds_num: number of samples per dataset, split by comma or int

echo ARG_MAX: `getconf ARG_MAX`

echo HTTP_PROXY: $http_proxy
echo HTTPS_PROXY: $https_proxy

echo PATH: $PATH
echo LD_LIBRARY_PATH: $LD_LIBRARY_PATH

source .venv/bin/activate

echo Hostname: $(hostname)
echo Working Directory: $(pwd)
nvidia-smi

################################################################################################
##################################### ANALYSIS STEP ############################################
OUTPUT_DIR=outputs/aware/LLaVA-7B-AWARe-${ANALYSIS}
SUBDIR=''
DS_OUTPUT_PATH=analyse_ds.jsonl
TARGET_DS=$TARGET_DS
EACH_DS_NUM=$EACH_DS_NUM
TARGET=$TARGET

SUBDIR="${EACH_DS_NUM//,/_}_$(echo $TARGET | sed 's/\*\.//g; s/\([^,]\)[^,]*,*/\1/g')"

bash scripts/AWARe/analyse.sh $OUTPUT_DIR $SUBDIR $DS_OUTPUT_PATH $TARGET_DS $EACH_DS_NUM $TARGET
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
    
bash scripts/AWARe/train.sh $DATASET_NAME $MODEL_SAVE_PATH $POS_PATH $NUM_TRAIN_EPOCHS
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