import argparse
import json
from pathlib import Path

from numpy.random import SeedSequence

from .aggregate_datasets import load_single_dataset

def get_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument(
        "--dataset_ratios_path",
        type=Path,
        required=True,
        help="path to JSON file containing input dataset ratios. Values ares dictionary: {'dataset_path': str, 'is_catalogue': bool, 'ratio': float}",
    )
    parser.add_argument("--split", type=str, default="train", help="split name, default 'train'")
    parser.add_argument(
        "--num_proc", type=int, default=1, help="number of procs to use for loading datasets, default 1"
    )
    # Parse args
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    with open(args.dataset_ratios_path, "f") as fi:
        ratios = json.load(fi)
    seed = SeedSequence(42)
    seed = seed.spawn(len(ratios))

    ds_ratio = ratios[args.dataset_index]
    load_single_dataset((ds_ratio, args.split, seed, args.num_proc))

if __name__ == "__main__":
    main()