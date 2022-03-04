def remove_lines_with_curly_brackets(examples):
    """Removes lines containing a '{' or '}' from the texts."""
    return {
        **examples,
        "text": [
            "\n".join([line for line in text.split("\n") if "{" not in line and "}" not in line])
            for text in examples["text"]
        ]
    }
