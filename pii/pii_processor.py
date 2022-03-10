import multiprocessing
import json
from functools import partial
from bigscience_pii_detect_redact import run_pii

path_jsonl = "/home/lucile/data_pii/pre_processing/lm_eu_oscar.jsonl"
save_path = "/home/lucile/data_pii/post_processing/lm_eu_oscar.jsonl"
lang = path_jsonl.split("/")[-1].replace("indic-", "").replace("lm_", "")[:2]

with open(path_jsonl, "r") as fi:
    jsonlines = fi.readlines()

def func_pii_multiprocessing(line, lang, save_path):
    line = json.loads(line)
    line["text"] = run_pii(line["text"], lang)
    with open(save_path, "a") as fi:
        fi.write(f"{json.dumps(line)}\n")

p = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
async_result = p.map_async(partial(func_pii_multiprocessing, lang=lang, save_path=save_path), jsonlines)
p.close()
p.join()
