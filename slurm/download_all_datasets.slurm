#!/bin/bash
#SBATCH --job-name=download_all_catalogue_datasets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/download_all_catalogue/%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CATALOGUE_DATA_REPO=$WORK/code/big_science/catalogue_data
pushd $CATALOGUE_DATA_REPO

DATASET_NAME="'bigscience-catalogue-lm-data/lm_en_arxiv', download_mode='force_redownload'"

echo "Downloading ${DATASET_NAME}"

module load git-lfs
module load unrar
python -c "from datasets import load_dataset; load_dataset($DATASET_NAME, use_auth_token=True)"
