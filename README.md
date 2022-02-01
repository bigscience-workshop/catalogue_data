# catalogue_data
Scripts to prepare catalogue data.

## Setup
Clone this repo.

Install git-lfs: https://github.com/git-lfs/git-lfs/wiki/Installation
```shell
sudo apt-get install git-lfs
git lfs install
```

Install dependencies:
```shell
sudo apt-add-repository non-free
sudo apt-get update
sudo apt-get install unrar
```

Create virtual environment, activate it and install dependencies:
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create User Access Token (with write access) at Hugging Face Hub: https://huggingface.co/settings/token
and set environment variables in the `.env` file at the root directory:
```
HF_USERNAME=<Replace with your Hugging Face username>
HF_TOKEN=<Replace with your Hugging Face API token>
GIT_USER=<Replace with your Git user>
GIT_EMAIL=<Replace with your Git email>
```

## Create metadata
To create dataset metadata (in file `dataset_infos.json`) run:
```shell
python create_metadata.py --repo <repo_id>
```
where you should replace `<repo_id>`, e.g. `bigscience-catalogue-lm-data/lm_ca_viquiquad`
