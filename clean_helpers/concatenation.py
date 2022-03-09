from datasets import Dataset

def concatenate_lm_fr_ester(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    ids = set([meta["id"].split("_id_")[0] for meta in ds["meta"]])
    
    new_texts = []
    new_meta = []
    for id_source in ids:
        ds_tmp = ds.filter(lambda exs: [id_source == meta["id"] for meta in exs["meta"]], batched=True)
        new_texts.append("\n".join(ds_tmp["text"]))
        new_meta.append({"id":id_source})
    
    new_ds = Dataset.from_dict({"text": new_texts, "meta": new_meta})
    return new_ds