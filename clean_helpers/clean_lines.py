from typing import List

def build_bad_substring_remover(bad_strings: List[str]):
    def remove_bad_substring(batch):
        return {
            **batch,
            "text": [
                "\n".join([line for line in text.split("\n") if not any([bs in line for bs in bad_strings])])
                for text in batch["text"]
            ]
        }
    return remove_bad_substring

def build_short_line_remover(min_length: int):
    def remove_short_lines(batch):
        return {
            **batch,
            "text": [
                "\n".join([line for line in text.split("\n") if len(line.split(" ")) < min_length])
                for text in batch["text"]
            ]
        }
    return remove_short_lines
