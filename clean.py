import os
import argparse
import logging
import random
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from functools import partial

from datasets.utils.logging import set_verbosity_info
from clean_helpers import build_small_docs_filter, filter_wiki_non_text_type, filter_wiki_user_titles, replace_newline_with_space

set_verbosity_info()
logger = logging.getLogger(__name__)


# Map functions
MAPS = {
    "replace_newline_with_space": replace_newline_with_space
}
# Filter functions
FILTERS = {
    "filter_wiki_user_titles": filter_wiki_user_titles,
    "filter_wiki_non_text_type": filter_wiki_non_text_type,
    "filter_small_docs": build_small_docs_filter(15),
}

assert set(MAPS.keys()).isdisjoint(set(FILTERS.keys()))

def revert_bool_output(examples, filter_function):
    booleans = filter_function(examples)
    return [not boolean for boolean in booleans]

def filter_diff_text(examples, in_text_col, out_text_col):
    return [text_in!=text_out for text_in, text_out in zip(examples[in_text_col], examples[out_text_col])]
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--maps-and-filters", nargs="+", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--checks-save-path", type=str, default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--load-arrow-file", action="store_true")
    args = parser.parse_args()
    return args

def apply_function(function_name: str, ds: Dataset, num_proc: int, batch_size: int, save_checks: bool) -> Dataset:
    if function_name in MAPS:
        in_text_col_map, out_text_col_map = "old_text", "text"
        map_function = MAPS[function_name]
        mapped_ds = ds.map(
                map_function, 
                batched=True, 
                num_proc=num_proc, 
                batch_size=batch_size
            )
        logger.info(f"Applied map function: {function_name}")
        if save_checks:
            remove_columns = set(ds.column_names)
            remove_columns.remove("text")
            ds = ds. remove_columns(remove_columns )
            ds = ds.rename_column("text", "old_text")
            mapped_diff_ds = concatenate_datasets([mapped_ds, ds], axis=1).filter(
                partial(filter_diff_text, in_text_col=in_text_col_map, out_text_col=out_text_col_map),
                batched=True, 
                num_proc=num_proc, 
                batch_size=batch_size
            )
            logger.info(f"     Initial number of samples: {len(ds)} samples")
            logger.info(f"     Modified samples: {len(ds) - len(mapped_diff_ds)} samples")
            logger.info(f"     Modified percentage: {(len(ds) - len(mapped_diff_ds)) / len(ds) * 100:.2f} %")
            idx_samples = random.sample(range(len(mapped_diff_ds)), min(len(mapped_diff_ds), 10))
            logger.info("Examples of modified examples:")
            for idx in idx_samples:
                logger.info(f"     Examples n°{idx} :\n{mapped_diff_ds[idx]}")
            return mapped_ds, mapped_diff_ds
        else:
            return mapped_ds, None
    elif function_name in FILTERS:
        filter_function = FILTERS[function_name]
        filtered_ds = ds.filter(filter_function, batched=True, num_proc=num_proc, batch_size=batch_size)
        logger.info(f"Applied filter: {function_name}")
        logger.info(f"     Initial number of samples: {len(ds)} samples")
        logger.info(f"     Removed samples: {len(ds) - len(filtered_ds)} samples")
        logger.info(f"     Removed percentage: {(len(ds) - len(filtered_ds)) / len(ds) * 100:.2f} %")
        if save_checks:
            filtered_out_ds = ds.filter(
                    partial(revert_bool_output, filter_function=filter_function), 
                    batched=True, num_proc=num_proc, 
                    batch_size=batch_size
                )
            idx_samples = random.sample(range(len(filtered_out_ds)), min(len(filtered_out_ds), 10))
            logger.info("Examples of filtered out examples:")
            for idx in idx_samples:
                logger.info(f"     Examples n°{idx} of filtered out examples:\n{filtered_out_ds[idx]}")
            return filtered_ds, filtered_out_ds
        else:
            return filtered_ds, None
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
        ds = load_dataset(args.dataset_path, split="train", use_auth_token=True, ignore_verifications=True)

    # Apply series of maps and filters
    logger.info(f" ===== Applying transformations =====")
    for map_or_filter in args.maps_and_filters:
        ds, ds_out = apply_function(map_or_filter, ds, args.num_proc, args.batch_size, save_checks= args.checks_save_path is not None)
        if ds_out is not None and len(ds_out) != 0:
            saving_path = os.path.join(args.checks_save_path, f"{map_or_filter}_checks")
            logger.info(f" ===== Saving examples to check after {map_or_filter}  =====")
            ds_out.save_to_disk(saving_path)


    # Save to disk
    logger.info(f" ===== Saving dataset =====")
    logger.info(f"Saving to json format at {args.save_path}.")
    ds.to_json(
        args.save_path,
        num_proc=args.num_proc
    )


if __name__ == "__main__":
    main()
