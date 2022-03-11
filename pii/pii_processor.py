import argparse
from functools import partial
from pathlib import Path
import logging

from datasets.utils.logging import set_verbosity_info
from datasets import load_dataset, load_from_disk

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--load_from_disk', action="store_true")
    parser.add_argument('--dataset_path', type=Path)
    parser.add_argument('--save_path', type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    return args

def run_pii(exs, lang):
    """
    Runs the given set of regexes on the data "lines" and pulls out the
    tagged items.
    The lines structure stores the language type(s). This can be used for
    language-specific regexes, although we're dropping that for now and using
    only "default"/non-language-specific regexes.
    """
    regex_metadata = []
    org_text = []
    new_text = []
    modified = []
    for text in exs["text"]:
        # What is this for...?
        text = text.encode().decode()
        matches = detect_pii(text, lang, high_risk_tags)
        if len(matches) > 0:
            # !!! REDACTION HAPPENS HERE !!!
            redacted_str, metadata = redact_pii(text, matches)
            regex_metadata.append(metadata)
            org_text.append(text)
            new_text.append(redacted_str)
            modified.append(True)
        else:
            regex_metadata.append([])
            org_text.append(text)
            new_text.append(text)
            modified.append(False)
    return {
        "regex_metadata": regex_metadata,
        "org_text": org_text,
        "new_text": new_text,
        "modified": modified
    }

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    logger.info(f" ===== Loading {args.dataset_path} =====")
    if args.load_from_disk:
        ds = load_from_disk(args.dataset_path)
    else:
        ds = load_dataset(args.dataset_path, data_files=["*.jsonl"], split="train")
    
    lang = args.dataset_path.split("/")[-1].replace("indic-", "").replace("lm_", "")[:2]

    logger.info(f" ===== Applying PII =====")
    ds = ds.map(
        partial(run_pii, lang=lang),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )
    
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
    logger.info(f" ===== Finish successfully =====")