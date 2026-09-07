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

    print("Creating BEIR dataset for:", split)
    tag = split.replace("_", "-")
    tag = tag.replace(".", "/")
    try:
        d = ir_datasets.load(tag + '/test')
    except:
        try:
            d = ir_datasets.load(tag + '/v2')
        except:
            d = ir_datasets.load(tag)

    dataset = tag.split("/")[-1]
    run = load_run_or_qrel(f"runs/run.beir.bm25-multifield.{dataset}.txt", topk=100)

    query = {}
    for q in d.queries_iter():
        query[q.query_id] = q.text

    qrel = {}
    for qid, docid, r, iteration in d.qrels_iter():
        if qid not in qrel:
            qrel[qid] = {}
        if int(iteration) == 0:
            if r > 0 :
                qrel[qid][docid] = r  #

    dataset_dict[split] = []
    for qid in qrel:

        positive_document_ids = [docid for docid, relevance in qrel[qid].items() if docid in run[qid]] # intersect
        dataset_dict[split].append({
            'query_id': qid, 
            'query_text': query[str(qid)],
            'positive_document_ids': positive_document_ids,
            'negative_document_ids': [docid for docid, relevance in run[qid].items() if docid not in qrel[qid]],
            'answer': None,
            'source': 'beir-bm25top100',
            'relevances': [qrel[qid][docid] for docid in positive_document_ids]
        })


## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/beir-subset")
