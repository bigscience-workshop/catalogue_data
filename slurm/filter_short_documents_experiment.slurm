#!/bin/bash
#SBATCH --job-name=filter_short_document_all_catalogue_datasets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/filter_short_document_catalogue_1/%x-%j.out           # output file name
#SBATCH --array=0-516   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

CATALOGUE_DATA_REPO=$WORK/code/big_science/catalogue_data
pushd $CATALOGUE_DATA_REPO

# =======  GET DATASET AND FUNCTIONS ======

DATASET_ID=$SLURM_ARRAY_TASK_ID
DATASET_TRAINING_CSV=$CATALOGUE_DATA_REPO/training.csv
readarray -t DATASET_PATH_AND_FUNCTIONS < <(python -c "
import pandas as pd
import numpy as np

data = pd.read_csv(\"${DATASET_TRAINING_CSV}\")
dataset = data.iloc[$DATASET_ID]
dataset = dataset.replace(np.nan, \"\", regex=True)
print(dataset[\"dataset_name\"])

print(
    dataset[\"--dedups\"] + " " +
    dataset[\"--maps-and-filters argument\"] + " " +
    "filter_small_docs_bytes_300 filter_small_docs_bytes_1024"
)

if dataset[\"--preprocessings\"] = \"\":
    print(dataset[\"--preprocessings\"])
else:
    print()
")

DATASET_PATH=${DATASET_PATH_AND_FUNCTIONS[0]}
PREPROCESSINGS=${DATASET_PATH_AND_FUNCTIONS[1]}
OFFICIAL_PREPROCESSINGS=${DATASET_PATH_AND_FUNCTIONS[2]}

if [[ $OFFICIAL_PREPROCESSINGS == KILL ]]
then
    echo "Not supposed to preprocess $DATASET_PATH"
    exit 0
fi


# ====== OBTAIN DATASET_NAME =======

if [[ $DATASET_PATH == /gpfswork/rech/six/uty16tp/dataset/tokenization/* ]]
then
    if [[ $DATASET_PATH == */data ]]
    then
        DATASET_NAME=${DATASET_PATH:48:-5}
    else
        DATASET_NAME=${DATASET_PATH:48}
    fi
else
    DATASET_NAME=$DATASET_PATH
fi

# ====== RUN PYTHON SCRIPT =======

BASE_PATH=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/filter/$DATASET_NAME
SAVE_PATH=$BASE_PATH/final
CHECKS_SAVE_PATH=$BASE_PATH/checks
LOGS_PATH=$BASE_PATH/logs.txt

mkdir -p $(dirname $SAVE_PATH)
mkdir -p $CHECKS_SAVE_PATH

export HF_DATASETS_OFFLINE=1

python clean.py \
    --dataset-path $DATASET_PATH \
    --preprocessings $PREPROCESSINGS \
    --save-path $SAVE_PATH \
    --checks-save-path $CHECKS_SAVE_PATH \
    --num-proc 10 \
    --batch-size 100 \
    --sampling-size-map-checks 1000 \
    --sampling-size-filter-checks 1000 \
    2>&1 | tee $LOGS_PATH
