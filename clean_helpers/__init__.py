from .filter_wiki_meta import filter_wiki_user_titles, filter_wiki_non_text_type
from .filter_small_docs_in_datasets import build_small_docs_filter
from .map_arabic import replace_newline_with_space
from .map_strip_substring import en_wiktionary_stripper
from .clean_lines import build_line_with_substring_remover
from .deduplication import build_dedup_template, dedup_document