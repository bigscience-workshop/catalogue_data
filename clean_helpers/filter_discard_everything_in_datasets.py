def filter_everything(examples):
    """Discard all documents"""
    return [False for text in examples["text"]]