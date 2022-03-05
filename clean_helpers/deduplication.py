from collections import defaultdict
from typing import List, Set, Tuple
import hashlib
import re
import string

from datasets import Dataset


# ======== DEDUPLICATION FUNCTIONS ===================

def build_dedup_template(min_template_line_size: int, min_template_line_occurence: int):
    def dedup_template(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
        """Computes and remove templates lines"""
        # Compute the hash of each lines
        split_into_lines_and_hashes = ds.map(
            split_text_to_lines_and_hash,
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=ds.column_names
        )
        lines_and_hashes = split_into_lines_and_hashes.remove_columns(
            set(split_into_lines_and_hashes.column_names) - {"lines", "hashes"}
        )

        # TODO: Batch read could help make it faster
        # Find template lines
        count_lines_occurence = defaultdict(lambda: 0)
        for row in lines_and_hashes:
            filtered_lines_and_hashes = [
                (line, hash_)
                for line, hash_ in zip(row["lines"], row["hashes"])
                if len(line) >= min_template_line_size
            ]
            for _, hash_ in filtered_lines_and_hashes:
                count_lines_occurence[hash_] += 1

        template_line_hashes = {k for k, v in count_lines_occurence.items() if v >= min_template_line_occurence}
        del count_lines_occurence

        # Clean dataset
        return split_into_lines_and_hashes.map(
            build_remove_template_lines(template_line_hashes),
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=split_into_lines_and_hashes.column_names
        )

    return dedup_template


def dedup_document(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    hashed_documents = ds.map(
        lambda batch: {**batch, "hash": get_hash_stripped(batch["text"])},
        num_proc=num_proc,
        batched=True,
        batch_size=batch_size,
    )

    hashes = set()

    def delete_text_from_duplicates(examples):
        examples = {"text": [text if is_new_hash(hash, hashes) else "" for text, hash in zip(examples["text"], examples["hash"])]}
        return examples

    return hashed_documents.map(
        delete_text_from_duplicates,
        num_proc=1,  # VERY IMPORTANT: hashes will be updated, and is not thread safe.
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
    ).remove_columns("hash")

    # return hashed_documents.filter(
    #     lambda hashes_: [is_new_hash(hash_, hashes) for hash_ in hashes_],
    #     num_proc=1,  # VERY IMPORTANT: hashes will be updated, and is not thread safe.
    #     input_columns=["hash"],
    #     batched=True,
    #     batch_size=batch_size,
    #     load_from_cache_file=False,
    # ).remove_columns("hash")


# =========== HELPERS ===============

# this only keeps letter characters
def get_hash_stripped(texts: List[str]) -> List[str]:
    """Get hash of content field."""
    stripped_texts = [re.sub(f'\s+|\d+|[{re.escape(string.punctuation)}]','', text) for text in texts]
    return [hashlib.md5(text.strip().encode("utf-8")).hexdigest() for text in stripped_texts]

# this doesn't, it just strips the whitespace
def get_hash(texts: List[str]) -> List[str]:
    """Get hash of content field."""
    return [hashlib.md5(text.strip().encode("utf-8")).hexdigest() for text in texts]


def split_text_in_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n")]


def split_text_to_lines_and_hash(batch):
    lines_per_texts = [split_text_in_lines(text) for text in batch["text"]]
    return {
        **{k: v for k, v in batch.items() if k != "text"},
        "lines": lines_per_texts,
        "hashes": [get_hash(lines) for lines in lines_per_texts]
    }


def clean_text(lines_and_hashes: List[Tuple[str, int]], template_line_hashes: Set[str]):
    return "\n".join([line for line, hash_ in lines_and_hashes if hash_ not in template_line_hashes])


def build_remove_template_lines(template_line_hashes: Set[str]):
    def remove_template_lines(batch):
        cleaned_texts = [
            clean_text(
                list(zip(lines, hashes)),
                template_line_hashes
            )
            for lines, hashes in zip(batch["lines"], batch["hashes"])
        ]
        return {
            **{
                key: value
                for key, value in batch.items()
                if key not in ["lines", "hashes"]
            },
            "text": [cleaned_text for cleaned_text in cleaned_texts]
        }

    return remove_template_lines


def is_new_hash(hash_: int, hashes: Set[int]) -> bool:
    """Check if current hash is still in set of unique hashes and remove if true."""
    if hash_ in hashes:
        return False
    else:
        hashes.add(hash_)
        return True
