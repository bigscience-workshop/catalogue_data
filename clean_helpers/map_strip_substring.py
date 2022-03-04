def build_substring_stripper(list_of_substrings):
    def strip_substrings(batch):
        text_batch = batch["text"]
        for substring in list_of_substrings:
            text_batch = [text.replace(substring, "") for text in text_batch]
        return {
            **batch,
            "text": text_batch
        }
    return strip_substrings
