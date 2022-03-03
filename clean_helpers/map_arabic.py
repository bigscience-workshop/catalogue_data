def replace_newline_with_space(batch, in_text_col, out_text_col):
    return {
        **batch,
        out_text_col: [text.replace("\n", " ") for text in batch[in_text_col]]
    }