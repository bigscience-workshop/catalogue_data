def build_small_docs_filter(min_word):
    def filter_small_docs(examples):
        """Discard documents with less than min_word words"""
        return [len(text.split(" ")) >= min_word for text in examples["text"]]
    return filter_small_docs
