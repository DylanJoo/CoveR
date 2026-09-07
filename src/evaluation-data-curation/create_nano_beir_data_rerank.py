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
    "nano_beir.nfcorpus",
    "nano_beir.nq",
    "nano_beir.quora",
    "nano_beir.scidocs",
    "nano_beir.scifact",
    "nano_beir.webis_touche2020",
]

for split in split_names:

    print("Creating BEIR dataset for:", split)
    tag = split.replace("_", "-")
    tag = tag.replace(".", "/")
    d = ir_datasets.load(tag)

    query = {}
    for qid, qtext in d.queries_iter():
        query[qid] = qtext

    document = {}
    for docid, doctext in d.docs_iter():
        document[docid] = doctext

    beirtag = tag.split('/')[-1]
    run = load_run_or_qrel(f"runs/run.beir.bm25-multifield.{beirtag}.txt", topk=100)

    dataset_dict[split] = []
    for qid in query:
        document_ids = [docid for docid in run[qid]]
        if len(document_ids) > 0:
            dataset_dict[split].append({
                'query_id': qid, 
                'query_text': query[str(qid)],
                'document_ids': document_ids,
                'source': 'nano-beir.bm25top100',
            })


## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/valid-nano-beir")
