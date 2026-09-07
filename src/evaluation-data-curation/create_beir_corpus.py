import os
import ir_datasets
from datasets import DatasetDict, Dataset, load_dataset

dataset_dict = {}
home_dir = os.path.expanduser("~/")

## QRELS from TREC
split_names = [
    "beir.arguana",
    "beir.climate_fever",
    "beir.dbpedia_entity",
    "beir.fever",
    "beir.fiqa",
    "beir.hotpotqa",
    "beir.nfcorpus",
    "beir.nq",
    "beir.quora",
    "beir.scidocs",
    "beir.scifact",
    "beir.trec_covid",
    "beir.webis_touche2020",
]

for split in split_names:

    print("Creating BEIR corpus for:", split)
    tag = split.replace("_", "-")
    tag = tag.replace(".", "/")
    d = ir_datasets.load(tag)

    document = {}
    for doc in d.docs_iter():
        docid = doc.doc_id
        doctext = doc.text
        doctitle = getattr(doc, 'title', '')
        document[docid] = (doctitle, doctext)

    dataset_dict[split] = []
    for docid in document:
        dataset_dict[split].append({
            'docid': docid,
            'title': document[docid][0],
            'text': document[docid][1],
            'source': 'beir',
        })


## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/beir-corpus")

load_dataset("DylanJHJ/beir-corpus")
