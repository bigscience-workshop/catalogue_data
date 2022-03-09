from .filter_wiki_meta import filter_wiki_user_titles, filter_wiki_non_text_type, filter_remove_empty_docs
from .filter_small_docs_in_datasets import build_small_docs_filter, build_small_docs_bytes_filter
from .map_arabic import replace_newline_with_space
from .map_strip_substring import en_wiktionary_stripper
from .map_remove_references import build_reference_remover
from .clean_lines import build_line_with_substring_remover
from .deduplication import build_dedup_template, build_dedup_document
from .sentence_splitter import build_sentence_splitter, sentence_split_langs