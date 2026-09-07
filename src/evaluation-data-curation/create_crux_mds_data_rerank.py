import os
from datasets import DatasetDict, Dataset, load_dataset
from crux.tools import load_run_or_qrel
from crux.tools.mds.ir_utils import load_topic

dataset_dict = {}
home_dir = os.path.expanduser("~/")

split_names = [
    "duc04",
    "multi_news",
]

ds_dict = load_dataset('DylanJHJ/crux-mds-corpus')
document = {}
for ds_split in ds_dict:
    for example in ds_dict[ds_split]:
        document[example['id']] = example['contents']

for split in split_names:

    print("Creating CRUX dataset for:", split)

    query = load_topic(subset=split)
    run = load_run_or_qrel(f"runs/bm25.crux-mds-{split}.txt", topk=100)

    dataset_dict[split] = []
    for qid in query:
        document_ids = [docid for docid in run[qid]]
        dataset_dict[split].append({
            'query_id': qid, 
            'query_text': query[str(qid)],
            'document_ids': document_ids,
            'source': 'crux-mds.bm25top100',
        })

## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/valid-crux-mds")
