def filter_wiki_non_text_type(examples):
    return [eval(meta)["type"] == "text" for meta in examples["meta"]]
