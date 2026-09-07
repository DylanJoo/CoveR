import os
import ir_datasets
from datasets import DatasetDict, Dataset
from crux.tools import load_run_or_qrel

dataset_dict = {}
home_dir = os.path.expanduser("~/")

## QRELS from TREC
split_names = [
    "nano_beir.arguana",
    "nano_beir.climate_fever",
    "nano_beir.dbpedia_entity",
    "nano_beir.fever",
    "nano_beir.fiqa",
    "nano_beir.hotpotqa",
    "nano_beir.msmarco",
    "nano_beir.nfcorpus",
    "nano_beir.nq",
    "nano_beir.quora",
    "nano_beir.scidocs",
    "nano_beir.scifact",
    "nano_beir.webis_touche2020",
]

for split in split_names:

    print("Creating BEIR corpus for:", split)
    tag = split.replace("_", "-")
    tag = tag.replace(".", "/")
    d = ir_datasets.load(tag)

    document = {}
    for docid, doctext in d.docs_iter():
        document[docid] = doctext

    dataset_dict[split] = []
    for docid in document:
        dataset_dict[split].append({
            'docid': docid,
            'title': "",
            'text': document[docid],
            'source': 'nano-beir',
        })


## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/nano-beir-corpus")
