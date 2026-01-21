#! /usr/bin/bash

# Output dir from analyse step
DIR_PATH=${1:-'outputs/aware/LLaVA-7B-AWARe'}
SUBDIR=${2:-'200_200_200_200_qkvm'}
ANALYSIS_FILE=${3:-'analysis.pt'}
QUOTA=${4:-10}
STRATEGY=${5:-'balanced'} # [balanced, global_highest]
SELECT_OUT_NAME=${6:-"${STRATEGY}_${QUOTA}"}

echo "Select args:"
echo "  DIR_PATH: $DIR_PATH"
echo "  SUBDIR: $SUBDIR"
echo "  ANALYSIS_FILE: $ANALYSIS_FILE"
echo "  QUOTA: $QUOTA"
echo "  STRATEGY: $STRATEGY"
echo "  OUTPUT_NAME: $SELECT_OUT_NAME"

python AWARe/select/select_node.py \
    --dir-path $DIR_PATH/$SUBDIR \
    --analysis-file $ANALYSIS_FILE \
    --quota $QUOTA \
    --strategy $STRATEGY \
    --output-name ${SELECT_OUT_NAME}.json