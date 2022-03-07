import argparse
import json
import multiprocessing

from datasets import load_dataset
from tqdm import tqdm


def get_size(name_dataset):
    try:
        dataset = load_dataset(name_dataset, use_auth_token=True, ignore_verifications=True, split="train")
        dataset = dataset.map(None, remove_columns=[column for column in dataset.column_names if column != "text"])
        print("Done for dataset:", name_dataset)

        len_bytes = 0
        for content in dataset["text"]:
            len_bytes += len(content.encode())
        return (name_dataset, len_bytes)
    except Exception as e:
        print(f"Failed for dataset: {name_dataset} because of {e}")
        return (name_dataset, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--dataset_list', type=str, default=None)
    parser.add_argument('--reuse_previous', action="store_true")
    args = parser.parse_args()

    list_datasets = [dataset_path.strip() for dataset_path in open(args.dataset_list).readlines()]
    print(len(list_datasets))

    if args.reuse_previous:
        previous_sizes = json.load(open("dataset_sizes.json"))
        list_datasets = [dataset_name for dataset_name in list_datasets if previous_sizes.get(dataset_name, 0) == 0]
    else:
        previous_sizes = None

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    async_result = p.imap_unordered(get_size, list_datasets)
    result = dict(tqdm(async_result))
    if previous_sizes is not None:
        result = dict(previous_sizes, **result)
    json.dump(result, open("dataset_sizes.json", "w"), ensure_ascii=False, indent=2)
    with open("dataset_sizes.csv", "w") as g:
        for dataset_name, size in sorted(result.items(), key=lambda x: x[0]):
            g.write(f"{dataset_name},{size * 1e-9:.4f}\n")
    p.close()
    p.join()
