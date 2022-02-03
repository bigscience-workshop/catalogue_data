import argparse
import json
import logging
import os
import multiprocessing
from contextlib import contextmanager
from functools import partial
from math import ceil
from pathlib import Path

from dotenv import load_dotenv
from numpy import log10
from numpy.random import default_rng, SeedSequence

from datasets import concatenate_datasets, load_dataset, utils, Features, Value

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument(
        "--dataset_ratios_path", type=str, required=True, help="path to JSON file containing input dataset ratios"
    )
    parser.add_argument("--split", type=str, default="train", help="split name, default 'train'")
    parser.add_argument(
        "--load_num_proc", type=int, default=1, help="number of procs to use for loading datasets, default 1"
    )
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


def load_datasets(args):
    ds_name, ratio, split, seed = args
    ds = load_dataset(ds_name, use_auth_token=os.environ["HF_USER_ACCESS_TOKEN"], ignore_verifications=True)
    if split not in ds:
        logger.info(f"No split named {split} in dataset {ds_name}")
        return
    ds = ds[split]
    # Process meta: add source_dataset and cast dict to str

    def process_meta(item, source_dataset=None):
        if "meta" not in item:
            item["meta"] = {}
        elif isinstance(item["meta"], str):
            item["meta"] = eval(item["meta"])
        item["meta"]["source_dataset"] = source_dataset
        item["meta"] = json.dumps(item["meta"])
        return item

    ds = ds.map(partial(process_meta, source_dataset=ds_name.split("/")[-1]))
    # Sample dataset
    if ratio != 1:
        rng = default_rng(seed)
        indices = rng.choice(len(ds), size=int(len(ds) * ratio), replace=False, shuffle=False)
        ds = ds.select(indices)
    return ds


def compute_number_of_shards(ds, max_size=10_000_000_000):
    if ds._indices is not None:
        ds_nbytes = ds.data.nbytes * len(ds._indices) / len(ds.data)
    else:
        ds_nbytes = ds.data.nbytes
    logger.info(f"Estimated dataset size: {ds_nbytes} bytes")
    logger.info(f"Max shard size: {max_size} bytes")
    number_shards = ceil(ds_nbytes / max_size)
    return number_shards if number_shards < len(ds) else len(ds)


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
    with tmp_path(save_path) as tmp_save_path:
        shard.to_json(
            tmp_save_path,
            num_proc=num_proc,
            batch_size=batch_size,
            compression="gzip",
        )


@contextmanager
def tmp_path(path):
    try:
        tmp_path = path.with_name(f"tmp-{path.name}")
        yield tmp_path
    except:
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.rename(path)


def main(
    dataset_ratios_path=None,
    split="train",
    seed=0,
    load_num_proc=1,
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
    seed = SeedSequence(seed)
    # Read dataset ratios
    with dataset_ratios_path.open() as f:
        dset_ratios = json.load(f)
    # Load datasets
    with multiprocessing.Pool(load_num_proc) as pool:
        dsets = [
            ds
            for ds in utils.tqdm(
                pool.imap(
                    load_datasets,
                    [
                        (ds_name, ratio, split, child_seed)
                        for (ds_name, ratio), child_seed in zip(dset_ratios.items(), seed.spawn(len(dset_ratios)))
                    ],
                ),
                total=len(dset_ratios),
                unit="ba",
                disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
                desc="Loading dataset",
            )
            if ds is not None
        ]
    if not dsets:
        logger.info(f"No datasets to be aggregated")
        return
    # Concatenate datasets
    dset = concatenate_datasets(dsets, split=split)
    # Shuffle
    dset = dset.shuffle(seed=seed)
    # Shard
    shards = shard_dataset(dset, max_size=shard_max_size)
    # Save
    save_shards(shards, path=save_path, num_proc=save_num_proc, batch_size=save_batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
