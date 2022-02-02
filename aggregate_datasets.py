import argparse
import json
import logging
import os
from math import ceil
from pathlib import Path

from dotenv import load_dotenv
from numpy import log10
from numpy.random import default_rng

from datasets import concatenate_datasets, load_dataset, Features, Value

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    # Datasets
    parser.add_argument(
        "--dataset_ratios_path", type=str, required=True, help="path to JSON file containing input dataset ratios"
    )
    parser.add_argument("--split", type=str, default="train", help="split name, default 'train'")
    # Shard
    parser.add_argument("--shard_max_size", type=int, default=10_000_000_000, help="max shard size, default 10GB")
    # Save
    parser.add_argument("--save_path", type=str, default=".", help="path to save the dataset, default '.'")
    parser.add_argument("--save_num_proc", type=int, default=1, help="number of procs to use for saving, default 1")
    parser.add_argument("--save_batch_size", type=int, help="batch size used for saving")
    # Parse args
    args = parser.parse_args()
    # Post-process args
    args.dataset_ratios_path = Path(args.dataset_ratios_path)
    args.save_path = Path(args.save_path)
    return args


def compute_number_of_shards(ds, max_size=10_000_000_000):
    if ds._indices is not None:
        ds_nbytes = ds.data.nbytes * len(ds._indices) / len(ds.data)
    else:
        ds_nbytes = ds.data.nbytes
    logger.info(f"Estimated dataset size: {ds_nbytes} bytes")
    logger.info(f"Max shard size: {max_size} bytes")
    return ceil(ds_nbytes / max_size)


def shard_dataset(ds, max_size=10_000_000_000):
    number_shards = compute_number_of_shards(ds, max_size=max_size)
    if number_shards <= 1:
        return [ds]
    shards = []
    logger.info(f"Shard dataset in {number_shards} shards")
    for shard_id in range(number_shards):
        logger.info(f"Shard {shard_id}/{number_shards}")
        shard = ds.shard(num_shards=number_shards, index=shard_id)
        shards.append(shard)
    return shards


def save_shards(shards, path=Path("."), num_proc=1, batch_size=None):
    path.mkdir(parents=True, exist_ok=True)
    num_shards = len(shards)
    for i, shard in enumerate(shards):
        save_dataset(shard, path=path, shard_id=i, num_shards=num_shards, num_proc=num_proc, batch_size=batch_size)


def save_dataset(shard, path=Path("."), shard_id=0, num_shards=1, num_proc=1, batch_size=None):
    width = int(log10(num_shards)) + 1
    save_path = path / f"shard-{shard_id:0>{width}}-of-{num_shards:0>{width}}.jsonl.gz"
    if save_path.exists():
        logger.info("Shard was already saved")
        return
    tmp_save_path = save_path.with_name(f"{save_path.name}.tmp")
    shard.to_json(
        tmp_save_path,
        num_proc=num_proc,
        batch_size=batch_size,
        compression="gzip",
    )
    tmp_save_path.rename(save_path)


def main(
    dataset_ratios_path=None,
    split="train",
    seed=0,
    shard_max_size=10_000_000_000,
    save_path=Path("."),
    save_num_proc=1,
    save_batch_size=None,
):
    # Init
    # Env variables
    if Path(".env").exists:
        load_dotenv()
    # Random generator
    rng = default_rng(seed=seed)
    # Read dataset ratios
    with dataset_ratios_path.open() as f:
        dset_ratios = json.load(f)
    # Load datasets
    dsets = []
    for ds_name, ratio in dset_ratios.items():
        ds = load_dataset(ds_name, use_auth_token=os.environ["HF_USER_ACCESS_TOKEN"], ignore_verifications=True)
        if split not in ds:
            logger.info(f"No split named {split} in dataset {ds_name}")
            continue
        ds = ds[split]
        # Cast meta dict to str
        if "meta" in ds.features and not isinstance(ds.features["meta"], Value):

            def convert_meta_to_str(item):
                item["meta"] = json.dumps(item["meta"])
                return item

            ds = ds.map(convert_meta_to_str)  # TODO: num_proc
        # Sample dataset
        if ratio != 1:
            indices = rng.integers(low=0, high=len(ds), size=int(len(ds) * ratio))
            ds = ds.select(indices)
        dsets.append(ds)
    if not dsets:
        logger.info(f"No datasets to be aggregated")
        return
    # Concatenate datasets
    dset = concatenate_datasets(dsets, split=split)
    # Shuffle
    dset = dset.shuffle(generator=rng)
    # Shard
    shards = shard_dataset(dset, max_size=shard_max_size)
    # Save
    save_shards(shards, path=save_path, num_proc=save_num_proc, batch_size=save_batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
