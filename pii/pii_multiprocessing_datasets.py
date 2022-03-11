import argparse
from bigscience_pii_detect_redact import run_pii
import json
from datasets import load_dataset
from multiprocessing import cpu_count


def parseArgs():
    parser = argparse.ArgumentParser(description="PII")
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
        default=1000,
        help="Number of documents to save for the metadata",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for the map function of datasets",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=-1,
        help="Number of processors for the map function of datasets. Default: maximum number of cores",
    )
    args = parser.parse_args()
    return args


def main():
    args = parseArgs()
    if args.num_proc == -1:
        args.num_proc = cpu_count()

    path_dataset_jsonl = args.path_dataset_jsonl
    path_save_dataset_jsonl = args.path_save_dataset_jsonl
    lang = path_dataset_jsonl.split("/")[-1].replace("indic-", "").replace("lm_", "")[:2]

    path_metadata_out_write = args.path_metadata_out_write
    list_metadata_out = []
    metadata_num_docs_to_write = args.metadata_num_docs_to_write


    def save_json(path_json_save, data):
        with open(path_json_save, 'w') as f:
            json.dump(data, f)


    def func_map(examples):
        examples_pii = examples
        for i, text in enumerate(examples["text"]):
            examples_pii["text"][i], metadata_out = run_pii(text, lang=lang)
            if len(list_metadata_out) < metadata_num_docs_to_write:
                if metadata_out:
                    list_metadata_out.append(metadata_out)
            if len(list_metadata_out) == metadata_num_docs_to_write:
                save_json(path_metadata_out_write, list_metadata_out)
        return examples_pii


    dataset = load_dataset('json', data_files=path_dataset_jsonl, split="train")
    dataset = dataset.map(func_map, batched=True, batch_size=args.batch_size, num_proc=args.num_proc)
    dataset.to_json(path_save_dataset_jsonl)


if __name__ == "__main__":
    main()
    
