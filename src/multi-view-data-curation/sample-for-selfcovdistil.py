import os
from tqdm import tqdm 
import argparse
import numpy as np
from datasets import DatasetDict, Dataset
from crux.tools import load_ratings, load_run_or_qrel
from crux.tools.researchy.ir_utils import load_topic, load_subtopics, load_request

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train', help='Dataset split to analyze')
parser.add_argument('--tau', type=int, default=4, help='Threshold for answerable subtopics')
args = parser.parse_args()

# main
split = args.split
tau = args.tau

topic = load_topic(split)
request = load_request(split)
subtopics = load_subtopics(split)
CRUX_ROOT = os.environ.get("CRUX_ROOT", '/users/judylan1/datasets/crux')
run = load_run_or_qrel(f'{CRUX_ROOT}/crux-researchy/runs/run.researchy-{split}-init-q.bm25+qwen3.clueweb22-b.txt')
judge = load_ratings(f'{CRUX_ROOT}/crux-researchy/judge/')

dataset_dict = {'pos_20.neg_51': []}
for qid in tqdm(run):

    # stat1: answerable 
    n = len(subtopics[qid])
    n_ans = (np.max([judge[qid][docid] for docid in judge[qid]], 0) >= tau).sum()
    n_unans = n - n_ans

    # stat2
    ## -1 mean exactly zero; -2 mean unjudged
    document_ids_all = [docid for docid in run[qid]]
    document_ids = {0.75: [], 0.5: [], 0.25: [], 0.0: [], -1: [], -2: []}
    for i, docid in enumerate(run[qid]):
        rating = (judge[qid][docid] or [0])
        c = sum(np.array(rating) >= tau) / n_ans if n_ans > 0 else 0

        if i < 20: # only include judged negative before 20
            for threshold in [0.75, 0.5, 0.25, 0.0]:
                if c >= threshold:
                    document_ids[threshold].append(docid)
                    if c == 0:
                        document_ids[-1].append(docid)
                    break

        if i > 50: # only include negative after 50
            document_ids[-2].append(docid)

    # 1. single-list sampling + coverage-based filtering
    filterd_top20 = [docid for docid in document_ids_all[:20] if docid not in document_ids[-1]]
    if len(filterd_top20) > 0:
        dataset_dict['pos_20.neg_51'].append({
            'query_id': qid, 
            'query_text': topic[str(qid)],
            'request_text': request[str(qid)],
            'positive_document_ids': filterd_top20,
            'negative_document_ids': document_ids_all[50:],
            'subquestions': subtopics[str(qid)],
            'answer': None,
            'source': f'crux-researchy.tau:{tau}.clueweb22-B',
        })

    # 3. coverage-based sampling with negative after 50
    for positive_category, negative_category in [
        ('high', 'zero'),
        ('high', 'low'),
        ('high', 'quarter'),
        ('high', 'half'),
        ('half', 'zero'),
        ('low', 'zero'),
        ('zero', 'high'),
    ]:
        tag = f"pos_{positive_category}.neg_{negative_category}"
        if tag not in dataset_dict:
            dataset_dict[tag] = []

        positive_docs = []
        if positive_category == 'high':
            positive_docs += document_ids[0.75]
        if positive_category == 'half':
            positive_docs += document_ids[0.5]
        if positive_category == 'low':
            positive_docs += document_ids[0.0]
        if positive_category == 'zero':
            positive_docs += document_ids[-1]

        negative_docs = []
        if negative_category == 'zero':
            negative_docs += document_ids[-1]
        if negative_category == 'low':
            negative_docs += document_ids[0.0]
        if negative_category == 'quarter':
            negative_docs += document_ids[0.25] + document_ids[0.0]
        if negative_category == 'half':
            negative_docs += document_ids[0.5] + document_ids[0.25] + document_ids[0.0]
        if negative_category == 'high':
            negative_docs += document_ids[0.75]

        ## remove redundant
        positive_docs = list(set(positive_docs))
        negative_docs = list(set(negative_docs))

        ## add to 16 negative
        negative_docs += document_ids[-2][:(16 - len(negative_docs))]

        if len(positive_docs) > 0:
            dataset_dict[tag].append({
                'query_id': qid, 
                'query_text': topic[str(qid)],
                'request_text': request[str(qid)],
                'positive_document_ids': positive_docs,
                'negative_document_ids': negative_docs,
                'subquestions': subtopics[str(qid)],
                'answer': None,
                'source': f'crux-researchy.tau:{tau}.clueweb22-B',
            })

## Transform to dataset (the base)
dataset_dict['train'] = dataset_dict['pos_20.neg_51']
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/crux-researchy-new")
