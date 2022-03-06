from typing import Dict


def parse_meta(meta) -> Dict:
    if isinstance(meta, str):
        meta = eval(meta)
    return meta
