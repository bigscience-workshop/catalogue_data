import argparse
import logging
import random
import sys
from functools import partial
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from pathlib import Path
from typing import Tuple, Optional, Callable
from datasets.utils.logging import set_verbosity_info
from clean_helpers import build_small_docs_filter, filter_wiki_non_text_type, filter_wiki_user_titles, \
    replace_newline_with_space, build_dedup_template, dedup_document, build_line_with_substring_remover, \
    en_wiktionary_stripper, build_small_docs_bytes_filter

set_verbosity_info()
logger = logging.getLogger(__name__)
from clean_helpers.filter_small_docs_in_datasets import build_small_docs_bytes_filter

# Map functions: function(batch: Dict) -> Dict
MAPS = {
    "replace_newline_with_space": replace_newline_with_space,
    "remove_lines_with_code": build_line_with_substring_remover(["{", "}", "[if", "<script"]),
    "remove_html_spans": build_line_with_substring_remover(["<span", "</span>", "<div", "</div>", "<a", "</a>", "br>"]),
    "remove_html_spans_sanad": build_line_with_substring_remover(["<img", "]]>", "<![CDATA", "//DW", "var ", "xtImg", "To view this video please enable JavaScript"]),
    "remove_wiki_mojobake": build_line_with_substring_remover(["À À"]),
    "strip_substrings_en_wiktionary": en_wiktionary_stripper
}
# Filter functions: function(batch: Dict) -> Dict
FILTERS = {
    "filter_remove_empty_docs": filter_wiki_user_titles,
    "filter_wiki_user_titles": filter_wiki_user_titles,
    "filter_wiki_non_text_type": filter_wiki_non_text_type,
    "filter_small_docs": build_small_docs_bytes_filter(500),
    ** {
        f"filter_small_docs_bytes_{i}": build_small_docs_bytes_filter(min_bytes=i) for i in [500, 1000, 7000]
    },
}
# Deduplication functions: function(ds: Dataset, num_proc: int, batch_size: int) -> Dataset
DEDUPS = {
    "dedup_template_soft": build_dedup_template(
        min_template_line_size=15,
        min_template_line_occurence=20,
    ),
    "dedup_pseudocrawl_newspapers": build_dedup_template(
        min_template_line_size=0,
        min_template_line_occurence=1000,
    ),
    "dedup_document": dedup_document
}

MAPS_KEYS = set(MAPS.keys())
FILTERS_KEYS = set(FILTERS.keys())
DEDUPS_KEYS = set(DEDUPS.keys())
assert MAPS_KEYS.isdisjoint(FILTERS_KEYS)
assert (MAPS_KEYS | FILTERS_KEYS).isdisjoint(DEDUPS_KEYS)

def quick_size_estimation(ds, content_key="text"):
    ds = ds.shuffle(1991)
    subset_size = min(10000, len(ds))
    ratio = float(len(ds)) / float(subset_size)
    partial_ds = ds.select(range(subset_size))
    return sys.getsizeof("".join(partial_ds[content_key])) * ratio

def revert_bool_output(examples, filter_function):
    booleans = filter_function(examples)
    return [not boolean for boolean in booleans]

def filter_diff_text(examples, in_text_col, out_text_col):
    return [text_in!=text_out for text_in, text_out in zip(examples[in_text_col], examples[out_text_col])]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--maps-and-filters", nargs="*", type=str, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--checks-save-path", type=Path, default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--load-arrow-file", action="store_true")
    parser.add_argument("--sampling-size-map-checks", type=int, default=None)
    parser.add_argument("--sampling-size-filter-checks", type=int, default=None)
    parser.add_argument("--from-scratch", action="store_true", help="Resave all datasets on disk.")
    parser.add_argument("--save-to-json", action="store_true", help="Save output dataset in json format.")
    return parser.parse_args()

def log_stats_samples(title: str, original_size: int, after_transformation_size: int, operation_type: str):
    logger.info(title)
    logger.info(f"     Initial number of samples: {original_size} samples")
    logger.info(f"     {operation_type} samples: {original_size - after_transformation_size} samples")
    logger.info(f"     {operation_type} percentage: {(original_size - after_transformation_size) / original_size * 100:.2f} %")
    logger.info(f"     Final number of samples: {after_transformation_size} samples")

def log_stats_bytes(title: str, original_size: float, after_transformation_size: float, operation_type: str):
    logger.info(title)
    logger.info(f"     Initial size in bytes: {original_size * 1e-9:.4f} GB")
    logger.info(f"     {operation_type} bytes: {(original_size - after_transformation_size) * 1e-9:.4f} GB")
    logger.info(f"     {operation_type} percentage: {(original_size - after_transformation_size) / original_size * 100:.2f} %")
    logger.info(f"     Final size in bytes: {after_transformation_size * 1e-9:.4f} GB")

def get_filtered_out_documents(
    ds: Dataset,
    filter_function: Callable,
    num_proc: int,
    batch_size: int,
    sampling_size: Optional[int]
) -> Dataset:
    filtered_out_ds = ds.filter(
        partial(revert_bool_output, filter_function=filter_function),
        batched=True, num_proc=num_proc,
        batch_size=batch_size
    )

    if sampling_size is not None:

        idx_samples = random.sample(range(len(filtered_out_ds)), min(len(filtered_out_ds), sampling_size))
        logger.info("Examples of filtered out examples:")
        for idx in idx_samples:
            logger.info(f"     Examples n°{idx} of filtered out examples:\n{filtered_out_ds[idx]}")
        filtered_out_ds = filtered_out_ds.select(idx_samples)

    return filtered_out_ds


def get_modified_documents(
    ds: Dataset,
    mapped_ds: Dataset,
    num_proc: int,
    batch_size: int,
    sampling_size: Optional[int],
) -> Dataset:
    remove_columns = set(ds.column_names)
    remove_columns.remove("text")
    ds = ds.remove_columns(remove_columns)
    ds = ds.rename_column("text", "old_text")

    assert len(mapped_ds) == len(ds), f"Mapping function are batched, but they should not alternate the size of the batch."
    mapped_diff_ds = concatenate_datasets([mapped_ds, ds], axis=1).filter(
        partial(filter_diff_text, in_text_col="old_text", out_text_col="text"),
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size
    )

    if sampling_size is not None:
        logger.info("Examples of modified examples:")
        idx_samples = random.sample(range(len(mapped_diff_ds)), min(len(mapped_diff_ds), sampling_size))
        for idx in idx_samples:
            logger.info(f"     Examples n°{idx} :\n{mapped_diff_ds[idx]}")
        mapped_diff_ds = ds.select(idx_samples)

    return mapped_diff_ds


def apply_function(function_name: str, ds: Dataset, args) -> Tuple[Dataset, Optional[Dataset]]:
    if function_name in MAPS:
        map_function = MAPS[function_name]
        mapped_ds = ds.map(
                map_function,
                batched=True,
                num_proc=args.num_proc,
                batch_size=args.batch_size,
                load_from_cache_file=False,
            )
        log_stats_bytes(f"Applied map function: {function_name}", len(ds), len(mapped_ds), operation_type="Modified")
        if args.checks_save_path is not None:
            mapped_diff_ds = get_modified_documents(ds, mapped_ds, args.num_proc, args.batch_size, args.sampling_size_map_checks)
            return mapped_ds, mapped_diff_ds
        else:
            return mapped_ds, None
    elif function_name in FILTERS:
        filter_function = FILTERS[function_name]
        filtered_ds = ds.filter(filter_function, batched=True, num_proc=args.num_proc, batch_size=args.batch_size)
        log_stats_samples(f"Applied filter: {function_name}", len(ds), len(filtered_ds), operation_type="Removed")
        if args.checks_save_path is not None:
            return filtered_ds, get_filtered_out_documents(ds, filter_function, args.num_proc, args.batch_size, args.sampling_size_filter_checks)
        else:
            return filtered_ds, None
    elif function_name in DEDUPS:
        dedup_function = DEDUPS[function_name]
        deduplicated_ds = dedup_function(ds, num_proc=args.num_proc, batch_size=args.batch_size)
        print(f"{len(ds)} samples in old ds")
        print(f"{quick_size_estimation(ds)} bytes in old ds")
        print(f"{len(deduplicated_ds)} samples in new ds")
        print(f"{quick_size_estimation(deduplicated_ds)} bytes in new ds")
        log_stats_bytes(f"Applied deduplication function: {function_name}", quick_size_estimation(ds), quick_size_estimation(deduplicated_ds), operation_type="Deduplicated")

        if args.sampling_size_map_checks is not None:
            logger.info("Examples of modified examples:")
            idx_samples = random.sample(range(len(deduplicated_ds)), min(len(deduplicated_ds), args.sampling_size_map_checks))
            for idx in idx_samples:
                logger.info(f"     Examples n°{idx} :\n{deduplicated_ds[idx]}")

        # Some deduplication do not preserve the number of samples, so alignement is lost. For example "dedup_document"
        if args.checks_save_path is not None:
            deduped_diff_ds = get_modified_documents(ds, deduplicated_ds, args.num_proc, args.batch_size, args.sampling_size_map_checks)
            return deduplicated_ds, deduped_diff_ds
        else:
            return deduplicated_ds, None
    else:
        raise NotImplementedError(f"{function_name} has not matched any existing function names. Available names:\n"
                                  f"Map functions: {MAPS_KEYS}\n"
                                  f"Filter functions: {FILTERS_KEYS}\n"
                                  f"Dedup functions: {DEDUPS_KEYS}\n"
                                  )

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
    for idx, map_or_filter in enumerate(args.maps_and_filters):
        ds, ds_diff = apply_function(map_or_filter, ds, args)
        if ds_diff is not None and len(ds_diff) != 0:
            saving_path = args.checks_save_path / f"{idx}_{map_or_filter}_checks"
            if not args.from_scratch and saving_path.exists():
                continue
            tmp_save_path = Path(saving_path.parent, f"tmp-{saving_path.name}")
            logger.info(f" ===== Saving examples to check after {map_or_filter}  =====")
            ds_diff.save_to_disk(tmp_save_path)
            tmp_save_path.rename(saving_path)


    # Save to disk
    if args.from_scratch or not args.save_path.exists():
        logger.info(f" ===== Saving dataset =====")
        logger.info(f"Saving to final dataset at {args.save_path}.")
        tmp_save_path = Path(args.save_path.parent, f"tmp-{args.save_path.name}")
        if len(ds) == 0:
            logger.info("Dataset was empty. Not saving anything.")
            return
        if args.save_to_json:
            ds.to_json(
                tmp_save_path,
                num_proc=args.num_proc
            )
        else:
            ds.save_to_disk(tmp_save_path)
        tmp_save_path.rename(args.save_path)


if __name__ == "__main__":
    main()
