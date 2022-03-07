#!/bin/bash
#SBATCH --job-name=filter_short_document_all_catalogue_datasets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/filter_short_document_catalogue_1/%x-%j.out           # output file name
#SBATCH --array=0-509   # TODO: modify according to the number of models you want to evaluated
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

language_to_short_filter_document = {
    \"ar\": 1000,
    \"ca\": 7000,
    \"code\": 7000,
    \"en\": 7000,
    \"es\": 7000,
    \"eu\": 0,
    \"fr\": 7000,
    \"id\": 500,
    \"indic-as\": 0,
    \"indic-bn\": 1000,
    \"indic-gu\": 500,
    \"indic-hi\": 1000,
    \"indic-kn\": 500,
    \"indic-ml\": 500,
    \"indic-mr\": 500,
    \"indic-ne\": 500,
    \"indic-or\": 0,
    \"indic-pa\": 500,
    \"indic-ta\": 500,
    \"indic-te\": 500,
    \"indic-ur\": 500,
    \"nigercongo\": 0,
    \"nigercongo-fon\": 0,
    \"nigercongo-ig\": 0,
    \"nigercongo-lg\": 0,
    \"nigercongo-rw\": 0,
    \"nigercongo-sw\": 0,
    \"nigercongo-wo\": 0,
    \"nigercongo-xh\": 0,
    \"nigercongo-yo\": 0,
    \"nigercongo-zu\": 0,
    \"pt\": 1000,
    \"vi\": 500,
    \"zh\": 7000,
    \"zht\": 7000,
}

import re

r = re.compile(r\"(?:/gpfswork/rech/six/uty16tp/dataset/tokenization/)?bigscience-catalogue-lm-data/lm_(.[^_]*)_*\")

data = pd.read_csv(\"${DATASET_TRAINING_CSV}\")
dataset = data.iloc[$DATASET_ID]

print(dataset[\"dataset_name\"])
print(\" \".join([f\"filter_small_docs_bytes_{i}\" for i in [500, 1000, 7000]]))

# language = r.match(dataset[\"dataset_name\"]).group(1)
# filter_min_length = language_to_short_filter_document[language]
# if filter_min_length > 0:
#     print(f\"filter_small_docs_bytes_{filter_min_length}\")
# else:
#     print()

if pd.notnull(dataset[\"--preprocessings\"]):
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
