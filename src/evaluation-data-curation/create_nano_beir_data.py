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

    qrel = {}
    for qid, docid, r, iteration in d.qrels_iter():
        if qid not in qrel:
            qrel[qid] = {}
        if int(iteration) == 0:
            qrel[qid][docid] = r  #

    dataset_dict[split] = []
    for qid in qrel:

        dataset_dict[split].append({
            'query_id': qid, 
            'query_text': query[str(qid)],
            'positive_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance > 0],
            'negative_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance == 0],
            'answer': None,
            'source': 'nano-beir',
            'relevances': [judge for docid, judge in qrel[qid].items() if judge > 0]
        })


## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/nano-beir")
