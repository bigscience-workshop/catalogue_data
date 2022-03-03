def replace_newline_with_space(batch):
    return {
        **batch,
        "text": [text.replace("\n", " ") for text in batch["text"]]
    }