import argparse
import json
import logging
import os
import multiprocessing
from contextlib import contextmanager
from functools import partial
from math import ceil
from pathlib import Path

import datasets
from dotenv import load_dotenv
from numpy import log10
from numpy.random import default_rng, SeedSequence

from datasets import concatenate_datasets, load_dataset, utils, Features, Value, Dataset

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument(
        "--dataset_ratios_path",
        type=str,
        required=True,
        help="path to JSON file containing input dataset ratios. Values ares dictionary: {'dataset_path': str, 'is_catalogue': bool, 'ratio': float}",
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


def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        return getattr(datasets, features["_type"])(features["dtype"])
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]


def get_features():
    features = {
        "HtmlPreprocessor_error": {"dtype": "int64", "id": None, "_type": "Value"},
        "HtmlPreprocessor_error_comment": {"dtype": "string", "id": None, "_type": "Value"},
        "content_languages": {"dtype": "string", "id": None, "_type": "Value"},
        "content_mime_detected": {"dtype": "string", "id": None, "_type": "Value"},
        "depth": {"dtype": "int16", "id": None, "_type": "Value"},
        "download_exception": {"dtype": "string", "id": None, "_type": "Value"},
        "external_urls": [{"dtype": "string", "id": None, "_type": "Value"}],
        "fetch_redirect": {"dtype": "string", "id": None, "_type": "Value"},
        "fetch_status": {"dtype": "int32", "id": None, "_type": "Value"},
        "fetch_time": {"dtype": "timestamp[ns]", "id": None, "_type": "Value"},
        "html_error": {"dtype": "string", "id": None, "_type": "Value"},
        "html_footer": [{"dtype": "string", "id": None, "_type": "Value"}],
        "html_head": [{"dtype": "string", "id": None, "_type": "Value"}],
        "html_str": {"dtype": "string", "id": None, "_type": "Value"},
        "html_title": [{"dtype": "string", "id": None, "_type": "Value"}],
        "metadata_html": [
            {
                "char_end_idx": {"dtype": "int64", "id": None, "_type": "Value"},
                "char_start_idx": {"dtype": "int64", "id": None, "_type": "Value"},
                "html_attrs": {
                    "attrs": [{"dtype": "string", "id": None, "_type": "Value"}],
                    "values": [{"dtype": "string", "id": None, "_type": "Value"}],
                },
                "key": {"dtype": "string", "id": None, "_type": "Value"},
                "relative_end_pos": {"dtype": "int64", "id": None, "_type": "Value"},
                "relative_start_pos": {"dtype": "int64", "id": None, "_type": "Value"},
                "type": {"dtype": "string", "id": None, "_type": "Value"},
                "value": {"dtype": "string", "id": None, "_type": "Value"},
            }
        ],
        "seed_id": {"dtype": "int32", "id": None, "_type": "Value"},
        "text": {"dtype": "string", "id": None, "_type": "Value"},
        "url": {"dtype": "string", "id": None, "_type": "Value"},
        "url_host_name": {"dtype": "string", "id": None, "_type": "Value"},
        "url_host_registered_domain": {"dtype": "string", "id": None, "_type": "Value"},
        "url_host_tld": {"dtype": "string", "id": None, "_type": "Value"},
        "url_surtkey": {"dtype": "string", "id": None, "_type": "Value"},
        "warc_filename": {"dtype": "string", "id": None, "_type": "Value"},
        "warc_record_length": {"dtype": "int32", "id": None, "_type": "Value"},
        "warc_record_offset": {"dtype": "int32", "id": None, "_type": "Value"},
    }
    return Features(convert_types(features))


def collapse_meta_(batch):
    """{"text": str, "meta": str}"""
    # TODO: check that
    columns_not_in_meta = ["text", "html_error", "html_footer", "html_head", "html_str", "html_title", "metadata_html"]
    columns_to_collapse = [name for name in batch.keys() if name not in columns_not_in_meta]

    number_of_rows = len(batch["text"])
    metas = [
        {
            **{name: batch[name][i] for name in columns_to_collapse},
            "source_dataset": f"pseudo-crawl--{batch['seed_id'][i]}",
        }
        for i in range(number_of_rows)
    ]

    new_batch = {"text": batch["text"], "meta": [str(meta) for meta in metas]}
    return new_batch


def collapse_meta(ds: Dataset, num_proc: int):
    """{"text": str, "meta": str}"""
    columns_to_keep = ["text"]
    column_names_to_remove = [name for name in ds.column_names if name not in columns_to_keep]
    return ds.map(collapse_meta_, batched=True, num_proc=num_proc, remove_columns=column_names_to_remove)


def load_datasets(args):
    try:
        ds_ratio, split, seed = args
        ds_name = ds_ratio["dataset_path"]
        ratio = ds_ratio["ratio"]
        is_catalogue = ds_ratio["is_catalogue"]
        # Load
        if is_catalogue:
            ds = load_dataset(ds_name, use_auth_token=True, ignore_verifications=True)
        else:
            # We assume it comes from pseudo crawl.
            # Pseudo crawl needs to be downloaded locally beforehand.
            features = get_features()
            dataset_path = Path(ds_name)
            ds = load_dataset(
                str((dataset_path / "text__html").absolute()), data_files="**.jsonl.gz", features=features
            )
        # Split
        if split not in ds:
            logger.info(f"No split named {split} in dataset {ds_name}")
            return
        ds = ds[split]

        # Sample dataset
        if ratio < 1:
            rng = default_rng(seed)
            indices = rng.choice(len(ds), size=int(len(ds) * ratio), replace=False, shuffle=False)
            ds = ds.select(indices)

        # Process meta: add source_dataset and cast dict to str
        if is_catalogue:

            def process_meta(item, source_dataset=None):
                if "meta" not in item:
                    item["meta"] = {}
                elif isinstance(item["meta"], str):
                    item["meta"] = eval(item["meta"])
                try:
                    item["meta"]["source_dataset"] = source_dataset
                except:
                    raise ValueError(f"Got {item['meta']} of type {type(item['meta'])}. Expected an dictionary. This is from {source_dataset}")
                item["meta"] = str(item["meta"])
                return item

            ds = ds.map(partial(process_meta, source_dataset=ds_name.split("/")[-1]))
        else:
            # collapse all meta data in "meta" column
            ds = collapse_meta(ds, num_proc=1)

        return ds
    except BaseException as err:
        logger.error(f"Error while loading dataset {ds_name}")
        raise err


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


def save_dataset(shard: Dataset, path=Path("."), shard_id=0, num_shards=1, num_proc=1, batch_size=None):
    width = int(log10(num_shards)) + 1
    save_path = path / f"shard-{shard_id:0>{width}}-of-{num_shards:0>{width}}.jsonl.gz"
    if save_path.exists():
        logger.info(f"Shard was already saved: {save_path}")
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
    logger.info("Start load_datasets")
    with multiprocessing.Pool(load_num_proc) as pool:
        dsets = [
            ds
            for ds in utils.tqdm(
                pool.imap(
                    load_datasets,
                    [
                        (dset_ratio, split, child_seed)
                        for dset_ratio, child_seed in zip(dset_ratios, seed.spawn(len(dset_ratios)))
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
    logger.info("Start concatenate_datasets")
    dset = concatenate_datasets(dsets, split=split)
    # Shuffle
    logger.info("Start shuffle dataset")
    dset = dset.shuffle(seed=seed)
    # Shard
    logger.info("Start shard_dataset")
    shards = shard_dataset(dset, max_size=shard_max_size)
    # Save
    logger.info("Start: save dataset")
    save_shards(shards, path=save_path, num_proc=save_num_proc, batch_size=save_batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
