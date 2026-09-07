import os
from tqdm import tqdm 
import argparse
import numpy as np
from datasets import Dataset, load_dataset
from crux.tools import load_run_or_qrel

# main
# run = load_run_or_qrel('/users/judylan1/datasets/msmarco-passage-kd/run.msmarco-v1-passage.bm25+qwen3-default.train.txt')
run = load_run_or_qrel('/users/judylan1/datasets/msmarco-passage-kd/run.all.txt', topk=2000)
ds = load_dataset('Tevatron/msmarco-passage-new')['train']
dataset_list = []

for example in ds:
    qid = example['query_id']
    query_text = example['query_text']

    positive_document_ids = [docid for docid in example['positive_document_ids']]
    negative_document_ids = [docid for docid in example['negative_document_ids']]

    dataset_list.append({
        'query_id': qid, 
        'query_text': query_text,
        'query_image': None,
        'positive_document_ids': positive_document_ids,
        'negative_document_ids': negative_document_ids,
        'positive_document_scores': [run[qid][docid] for docid in positive_document_ids],
        'negative_document_scores': [run[qid][docid] for docid in negative_document_ids],
        'answer': None,
        'source': 'msmarco-qwen3-0.6b-rerank'
    })

## Transform to dataset (the base)
dataset = Dataset.from_list(dataset_list)
dataset.push_to_hub("DylanJHJ/msmarco-passage-new-qwen3-0.6b-rerank")
