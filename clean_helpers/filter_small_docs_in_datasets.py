def build_small_docs_filter(min_word):
    def filter_small_docs(examples):
        """Discard documents with less than 15 words"""
        return [len(text.split(" ")) >= min_word for text in examples["text"]]
    return filter_small_docs

# in clean.py
FILTERS = {
   "filter_en_small_docs": filter_small_docs(15),
}
