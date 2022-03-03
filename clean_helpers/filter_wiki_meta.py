def filter_user_titles(examples):
    return [not eval(meta)["title"].startswith("User ") for meta in examples["meta"]]
