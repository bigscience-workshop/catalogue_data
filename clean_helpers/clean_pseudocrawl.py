def remove_lines_with_curly_brackets(examples):
    """Removes lines containing a '{', '}', "[if", or "<script" from the texts."""
    bad_strings = ["{", "}", "[if", "<script"]
    return {
        **examples,
        "text": [
            "\n".join([line for line in text.split("\n") if not any([bs in line for bs in bad_strings])])
            for text in examples["text"]
        ]
    }
