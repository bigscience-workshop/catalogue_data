import argparse
from datasets import load_dataset
from bigscience_pii_detect_redact import run_pii
from multiprocessing import cpu_count
import json


def parseArgs():
    parser = argparse.ArgumentParser(description="Filtering.")
    parser.add_argument(
        "path_dataset_jsonl",
        type=str,
        help="Path of the dataset in jsonl format to load",
    )
    parser.add_argument(
        "path_save_dataset_jsonl",
        type=str,
        help="Path to save the dataset in jsonl format after PII",
    )
    parser.add_argument(
        "path_metadata_out_write",
        type=str,
        help="Path to save the metadata of the PII",
    )
    parser.add_argument(
        "--metadata_num_docs_to_write",
        type=int,
        default=100000,
        help="Number of documents to save for the metadata",
    )
    args = parser.parse_args()
    return args


def main():
    args = parseArgs()

    path_dataset_jsonl = args.path_dataset_jsonl
    path_save_dataset_jsonl = args.path_save_dataset_jsonl
    lang = path_dataset_jsonl.split("/")[-1].replace("indic-", "").replace("lm_", "")[:2]

    path_metadata_out_write = args.path_metadata_out_write
    list_metadata_out = []
    metadata_num_docs_to_write = args.metadata_num_docs_to_write


    def func_map(examples):
        for example in examples:
            example["text"], metadata_out = run_pii(example["text"], lang=lang)
            if len(list_metadata_out) < metadata_num_docs_to_write:
                list_metadata_out.append(metadata_out)
        return examples


    def save_json(path_json_save, data):
        with open(path_json_save, 'w') as f:
            json.dump(data, f)


    dataset = load_dataset('json', data_files=path_dataset_jsonl)
    dataset = dataset.map(func_map, batched=True, num_proc=cpu_count())
    dataset.to_json(path_save_dataset_jsonl)
    save_json(path_metadata_out_write, list_metadata_out)


if __name__ == "__main__":
    main()
    
