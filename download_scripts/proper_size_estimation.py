import argparse
import json
import multiprocessing
import os

from datasets import load_dataset


def get_size(name_dataset):
    try:
        dataset = load_dataset(name_dataset, use_auth_token="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", ignore_verifications=True, split="train").map(None, remove_columns=["meta"])
        path = f"dumps/{name_dataset}.jsonl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dataset.to_json(path)
        print("Done for dataset:", name_dataset)
        return (name_dataset, os.path.getsize(path))
    except Exception as e:
        print(f"Failed for dataset: {name_dataset} because of {e}")
        return (name_dataset, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--ratio_file', type=str, default=None)
    args = parser.parse_args()

    f = open(args.ratio_file, "r")
    ratios = json.load(f)
    list_datasets = [item["dataset_path"] for item in ratios]
    f.close()

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    async_result = p.imap_unordered(get_size, list_datasets)
    result = dict(async_result)
    json.dump(result, open("dataset_sizes.json", "w"), ensure_ascii=False, indent=2)
    p.close()
    p.join()
