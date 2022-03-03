def filter_wiki_user_titles(examples):
    return [not eval(meta)["title"].startswith("User ") for meta in examples["meta"]]

def filter_wiki_non_text_type(examples):
    return [eval(meta)["type"] == "text" for meta in examples["meta"]]

