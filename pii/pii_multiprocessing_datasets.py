from datasets import load_dataset
from bigscience_pii_detect_redact import run_pii
from multiprocessing import cpu_count
import json


path_metadata_out_write = "metadata_out.json"
list_metadata_out = []
metadata_num_docs_to_write = 100000

path_dataset_json = "dataset.jsonl"
path_save_dataset = "dataset_pii.jsonl"


def func_map(examples):
    for example in examples:
        example["text"], metadata_out = run_pii["text"]
        if len(list_metadata_out) < metadata_num_docs_to_write:
            list_metadata_out.append(metadata_out)
    return examples


def save_json(path_json_save, data):
    with open(path_json_save, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    dataset = load_dataset('json', data_files=path_dataset_json)
    dataset = dataset.map(func_map, num_proc=cpu_count())
    dataset.to_json(path_save_dataset)
    save_json(path_metadata_out_write, list_metadata_out)
    

