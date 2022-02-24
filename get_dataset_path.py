import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument("--dataset-ratios-file", type=Path)
    parser.add_argument("--index", type=int)
    # Parse args
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    with open(args.dataset_ratios_file, "r") as fi:
        data = json.load(fi)

    assert args.index < len(data)
    print(data[args.index])


if __name__ == "__main__":
    main()