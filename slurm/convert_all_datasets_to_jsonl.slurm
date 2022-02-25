#!/bin/bash
#SBATCH --job-name=convert_all_catalogue_to_jsonl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/convert_all_catalogue_to_jsonl/%x-%j.out           # output file name
#SBATCH --array=0-491   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CATALOGUE_DATA_REPO=$WORK/code/big_science/catalogue_data
pushd $CATALOGUE_DATA_REPO

# DATASET_PATH=${DATASETS_NAMES[$SLURM_ARRAY_TASK_ID]}
DATASET_PATH=$(python get_dataset_path.py --dataset-ratios-file training_dataset_ratios.json --index $SLURM_ARRAY_TASK_ID)
echo "Converting $DATASET_PATH to json"

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

NUM_SHARDS=8

SAVE_PREFIX=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/jsonl_"$NUM_SHARDS"_shards/$DATASET_NAME/data
mkdir -p $(dirname $SAVE_PREFIX)

echo "Downloading ${DATASET_NAME}"
export HF_DATASETS_OFFLINE=1


module load git-lfs
module load unrar
python -c "
from datasets import load_dataset
from pathlib import Path

ds = load_dataset(\"$DATASET_PATH\", use_auth_token=True, split=\"train\")

for shard_id in range($NUM_SHARDS):
    filepath = Path(f\"${SAVE_PREFIX}_{shard_id}_$NUM_SHARDS.jsonl\")
    if filepath.exists():
        continue
    shard = ds.shard(num_shards=$NUM_SHARDS,index=shard_id)
    tempfilepath = f"{str(filepath.absolute())}.tmp"
    shard.to_json(tempfilepath, num_proc=10)
    tempfilepath.rename(filepath)
"
