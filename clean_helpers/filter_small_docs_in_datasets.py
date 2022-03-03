def filter_small_docs(examples):
    """Discard documents with less than 15 words"""
    return [len(text.split(" ")) >= 15 for text in examples["text"]]
