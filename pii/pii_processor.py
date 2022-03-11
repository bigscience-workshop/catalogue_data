import multiprocessing
import json
from functools import partial
from bigscience_pii_detect_redact import run_pii

path_jsonl = "lm_eu_oscar.jsonl"
save_path = "lm_eu_oscar-redacted.jsonl"
metadata_save_path = "lm_eu_oscar-redacted-metadata.jsonl"

lang = path_jsonl.split("/")[-1].replace("indic-", "").replace("lm_", "")[:2]

with open(path_jsonl, "r") as fi:
    jsonlines = fi.readlines()

def func_pii_multiprocessing(line, lang, save_path, metadata_save_path):
    line = json.loads(line)

    match_set = run_pii(line["text"], lang)
    redacted_str, metadata = match_set
    line["text"] = redacted_str
    with open(save_path, "a") as fi:
        fi.write(f"{json.dumps(line)}\n")
    with open(metadata_save_path, "a") as fi:
        fi.write(f"{json.dumps(metadata)}\n")

if __name__ == '__main__':
    p = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    async_result = p.map_async(partial(func_pii_multiprocessing, lang=lang, save_path=save_path, metadata_save_path=metadata_save_path), jsonlines)
    p.close()
    p.join()
