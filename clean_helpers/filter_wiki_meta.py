def filter_wiki_non_text_type(examples):
    return [True if eval(meta)["type"] == "text" else False for meta in examples["meta"]]
