from typing import Optional, Dict, List

import pandas as pd
import re

from clean_helpers.utils import get_language

normalise_dataset_name_regex = re.compile(r"^(?:/gpfswork/rech/six/uty16tp/dataset/tokenization/)?(bigscience-catalogue-lm-data/[^/]+)(?:/data)?$")
def get_dedup_args(row: Dict) -> List[str]:
    ds_name = normalise_dataset_name_regex.match(row["dataset_name"]).group(1)
    if "pseudocrawl" in ds_name:
        list_of_dedups = ["dedup_document_on_url", "dedup_document", "dedup_pseudocrawl_newspapers"]
    else:
        list_of_dedups = ["dedup_document", "dedup_template_soft"]

    if all(black_list_ds not in ds_name for black_list_ds in ["open_subtitles", "europarl", "uncorpus"]):
        list_of_dedups += ["dedup_template_soft"]

    list_of_dedups += ["filter_remove_empty_docs"]
    return list_of_dedups

language_to_short_filter_document = {
    "ar": 1000,
    "ca": 7000,
    "code": 7000,
    "en": 7000,
    "es": 7000,
    "eu": 0,
    "fr": 7000,
    "id": 500,
    "indic-as": 0,
    "indic-bn": 1000,
    "indic-gu": 500,
    "indic-hi": 1000,
    "indic-kn": 500,
    "indic-ml": 500,
    "indic-mr": 500,
    "indic-ne": 500,
    "indic-or": 0,
    "indic-pa": 500,
    "indic-ta": 500,
    "indic-te": 500,
    "indic-ur": 500,
    "nigercongo-sw": 0,
    "nigercongo-yo": 0,
    "nigercongo-rw": 0,
    "nigercongo-xh": 0,
    "nigercongo-ig": 0,
    "nigercongo-zu": 0,
    "nigercongo-sn": 0,
    "nigercongo-lg": 0,
    "nigercongo-wo": 0,
    "nigercongo-rn": 0,
    "nigercongo-fon": 0,
    "nigercongo-nso": 0,
    "nigercongo-ln": 0,
    "nigercongo-tn": 0,
    "nigercongo-tw": 0,
    "nigercongo-ny": 0,
    "nigercongo-st": 0,
    "nigercongo-ts": 0,
    "nigercongo-ak": 0,
    "nigercongo-bm": 0,
    "nigercongo-ki": 0,
    "nigercongo-tum": 0,
    "pt": 1000,
    "vi": 500,
    "zhs": 7000,
    "zht": 7000,
}
def get_filter_on_small_documents_args(row: Dict) -> Optional[str]:
    language = get_language(row["dataset_name"])

    filter_min_length = language_to_short_filter_document[language]
    if filter_min_length > 0:
        return f"filter_small_docs_bytes_{filter_min_length}"
    else:
        return None

def main():
    data = pd.read_csv("training.csv")

    data["dedups"] = [" ".join(get_dedup_args(row[1])) for row in data.iterrows()]
    print(data[:5]["dedups"])

    data["filter_short_documents"] = [get_filter_on_small_documents_args(row[1]) for row in data.iterrows()]
    print(data[:5]["filter_short_documents"])

    data.to_csv("training_with_dedups.csv")
    
if __name__ == "__main__":
    main()