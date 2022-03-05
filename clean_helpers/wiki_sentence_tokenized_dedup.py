from tqdm import tqdm
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import multiprocessing


def init():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# This might need to be tuned and experimented with, potentially adapted for languages
def preprocess_article(article, dataset_name):
    """Preprocesses an article upstream of the punkt tokenizer."""
    if 'wiktionary' in dataset_name:
        return article.replace('./','.').replace('/','')
    elif 'wikiquote' in dataset_name:
        return article.replace(':','.').replace('-','')
    else:
        return article.replace(':','').replace('|',' ').replace('{','').replace('}','').replace("''","")

def get_lines_to_skip(dset, alpha=20, num_samples_to_check=10_000):
    """Looks at up to the first num_samples_to_check documents for a dataset and records lines that appear in at least alpha % of the unique pages"""
    num_range = min(num_samples_to_check, dset["train"].num_rows)
    line_counts = {}
    seen_pages = {}
    dset_sample = dset["train"].select(range(num_range))
    dset_name = dset['train'].config_name
    for page in tqdm(dset_sample):
        article = page["text"]
        article = preprocess_article(article, dset_name)
        if not seen_pages.get(article, False):
            seen_pages[article] = True
            for line in sent_tokenize(article):
                line_counts[line.strip()] = line_counts.get(line.strip(), 0) + 1
    thres_skip = max(10, len(seen_pages)* alpha // 100 )
    skip_dict = {line: ct for line, ct in line_counts.items() if ct > thres_skip}
    print("Total number of articles checked: ", num_range)
    print("Total number of lines: ", sum(line_counts.values()))
    return list(skip_dict.keys())

def map_filter_lines(element, skip_list=[]):
    """Mapper that filters out the sentences flagged by the get_lines_to_skip function"""
    article = element["text"]
    if skip_list:
        for to_skip in skip_list:
            article = article.replace(to_skip, '')
    return {"text": article}

def filter_wiki_dataset(dset):
    init()
    ds = load_dataset(dset, use_auth_token=True, ignore_verifications=True)
    skip_list_ds = get_lines_to_skip(ds)
    return ds.map(map_filter_lines(), fn_kwargs={"skip_list": skip_list_ds}, num_proc=multiprocessing.cpu_count() // 2)
