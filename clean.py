import argparse
import logging
from datasets import Dataset, load_dataset

from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


# Map functions
MAPS = {}
# Filter functions
FILTERS = {}

assert set(MAPS.keys()).isdisjoint(set(FILTERS.keys()))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--maps-and-filters",
        nargs="+",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save-dataset",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1
    )
    args = parser.parse_args()
    return args

# TODO: think of a batch mechanism
def apply_function(function_name: str, ds: Dataset, num_proc: int) -> Dataset:
    if function_name in MAPS:
        map_function = MAPS[function_name]
        return ds.map(map_function, num_proc=num_proc)
    elif function_name in FILTERS:
        filter_function = FILTERS[function_name]
        return ds.filter(filter_function, num_proc=num_proc)
    else:
        raise NotImplemented(f"{function_name} has not matched any existing function names. Available names:\n"
                             f"Map functions: {MAPS.keys()}\n"
                             f"Filter functions: {FILTERS.keys()}\n")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    # Load dataset
    # TODO: load_from_disk as well.
    ds = load_dataset(args.dataset_path)

    # Apply series of maps and filters
    for map_or_filter in args.maps_and_filters:
        ds = apply_function(map_or_filter, ds, args.num_proc)

    # Save to disk
    # TODO: define exporting strategy
    ds.save_to_disk(args.save_dataset)


if __name__ == "__main__":
    main()