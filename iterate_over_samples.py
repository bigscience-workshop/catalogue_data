from datasets import load_dataset
import regex as re

ds = load_dataset("bigscience-catalogue-lm-data/lm_indic-hi_indic_nlp_corpus", split="train", use_auth_token=True, ignore_verifications=True)

r = re.compile(r".*\d.*")

def remove_digits_documents(batch):
    return [r.match(text) is not None for text in batch["text"]]

def build_small_docs_filter(min_word):
    def filter_small_docs(examples):
        """Discard documents with less than min_word words"""
        return [len(text.split(" ")) >= min_word for text in examples["text"]]
    return filter_small_docs

filtered_ds = ds.filter(build_small_docs_filter(15), batched=True)

print(f"Kept proportions: {len(filtered_ds) / len(ds):.2%}")

for i, row in enumerate(filtered_ds):
    print(f"""
    Sample {i}
    
    {row["text"]}
    
    ________
    """)
    input()

# =====

ds = load_dataset("bigscience-catalogue-lm-data/lm_vi_binhvq_news_corpus", split="train", use_auth_token=True, ignore_verifications=True)

filtered_ds = ds

print(f"Kept proportions: {len(filtered_ds) / len(ds):.2%}")

for i, row in enumerate(filtered_ds):
    print(f"""
    Sample {i}

    {row["text"]}

    ________
    """)
    input()

python clean.py \
    --dataset-path bigscience-catalogue-lm-data/lm_pt_brwac \
    --maps-and-filters dedup \
    --save-path to_delete/lm_pt_brwac/final_v1 \
    --checks-save-path to_delete/lm_pt_brwac/cleaned \
    --num-proc 8 \
    --batch-size 100

python clean.py \
    --dataset-path bigscience-catalogue-lm-data/lm_pt_brwac \
    --maps-and-filters dedup_template_pt_bwarc \
    --save-path to_delete/lm_pt_brwac/final_v1 \
    --checks-save-path to_delete/lm_pt_brwac/cleaned \
    --num-proc 8 \
    --batch-size 100

for row in ds:
    print(f"{row['text']}")
    print()
    print(f"{row['old_text']}")
    print()
    print(" ================== ")
    print()
    input()