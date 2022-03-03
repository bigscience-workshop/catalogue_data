def remove_lines_with_curly_brackets(examples):
    """Removes lines containing a '{' from the texts."""
    fixed_texts = []
    for text in examples["text"]:
        fixed_lines = []
        for line in text.split("\n"):
            if "{" not in line:
                fixed_lines.append(line)
        fixed_texts.append("\n".join(fixed_lines))
    examples["text"] = fixed_texts
    return examples