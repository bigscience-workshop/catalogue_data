import argparse
import logging
from datasets import Dataset, load_dataset, load_from_disk

from datasets.utils.logging import set_verbosity_info
from clean_helpers import filter_wiki_non_text_type
from clean_helpers import filter_small_docs

set_verbosity_info()
logger = logging.getLogger(__name__)


# Map functions
MAPS = {}
# Filter functions
FILTERS = {
    "filter_wiki_non_text_type": filter_wiki_non_text_type,
    "filter_small_docs": filter_small_docs,
}

assert set(MAPS.keys()).isdisjoint(set(FILTERS.keys()))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--maps-and-filters", nargs="+", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--load-arrow-file", action="store_true")
    args = parser.parse_args()
    return args

def apply_function(function_name: str, ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    if function_name in MAPS:
        map_function = MAPS[function_name]
        mapped_function = ds.map(map_function, batched=True, num_proc=num_proc, batch_size=batch_size)
        logger.info(f"Applied map function: {function_name}")
        return mapped_function
    elif function_name in FILTERS:
        filter_function = FILTERS[function_name]
        filtered_ds = ds.filter(filter_function, batched=True, num_proc=num_proc, batch_size=batch_size)
        logger.info(f"Applied filter: {function_name}")
        logger.info(f"     Initial number of samples: {len(ds)} samples")
        logger.info(f"     Removed samples: {len(ds) - len(filtered_ds)} samples")
        logger.info(f"     Removed percentage: {(len(ds) - len(filtered_ds)) / len(ds) * 100:.2f} %")
        return filtered_ds
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
    logger.info(f" ===== Loading {args.dataset_path} =====")
    if args.load_arrow_file:
        ds = load_from_disk(args.dataset_path)
    else:
        ds = load_dataset(args.dataset_path, split="train", ignore_verifications=True)

    # Apply series of maps and filters
    logger.info(f" ===== Applying transformations =====")
    for map_or_filter in args.maps_and_filters:
        ds = apply_function(map_or_filter, ds, args.num_proc, args.batch_size)

    # Save to disk
    logger.info(f" ===== Saving dataset =====")
    logger.info(f"Saving to json format at {args.save_path}.")
    ds.to_json(
        args.save_path,
        num_proc=args.num_proc
    )


if __name__ == "__main__":
    main()
