import os
import ir_datasets
from datasets import DatasetDict, Dataset
from crux.tools import load_run_or_qrel

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
    "beir.webis_touche2020",
    "beir.trec_covid",
]

for split in split_names:

    print("Creating BEIR corpus for:", split)
    tag = split.replace("_", "-")
    tag = tag.replace(".", "/")
    d = ir_datasets.load(tag)

    dataset = tag.split("/")[-1]
    run = load_run_or_qrel(f"runs/run.beir.bm25-multifield.{dataset}.txt", topk=100)
    all_docids = {docid for qid in run for docid in run[qid].keys()}

    document = {}
    for doc in d.docs_iter():
        docid = doc.doc_id
        doctext = doc.text
        if docid in all_docids:
            document[docid] = doctext

    dataset_dict[split] = []
    for docid in document:
        dataset_dict[split].append({
            'docid': docid,
            'title': "",
            'text': document[docid],
            'source': 'beir-bm25top100',
        })


## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/beir-subset-corpus")
