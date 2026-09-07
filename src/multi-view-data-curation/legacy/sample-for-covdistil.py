import os
from tqdm import tqdm 
import argparse
import numpy as np
from datasets import DatasetDict, Dataset
from crux.tools import load_ratings, load_run_or_qrel
from crux.tools.researchy.ir_utils import load_topic, load_subtopics

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train', help='Dataset split to analyze')
parser.add_argument('--tau', type=int, default=4, help='Threshold for answerable subtopics')
args = parser.parse_args()

# main
split = args.split
tau = args.tau

topic = load_topic(split)
subtopics = load_subtopics(split)
CRUX_ROOT = os.environ.get("CRUX_ROOT", '/users/judylan1/datasets/crux')
run = load_run_or_qrel(f'{CRUX_ROOT}/crux-researchy/runs/run.researchy-{split}-init-q.bm25+qwen3.clueweb22-b.txt')
judge = load_ratings(f'{CRUX_ROOT}/crux-researchy/judge/')

dataset_dict = {'cov.pos_20.neg_51.f': []}
for qid in tqdm(run):

    # stat1: answerable 
    n = len(subtopics[qid])
    n_ans = (np.max([judge[qid][docid] for docid in judge[qid]], 0) >= tau).sum()
    n_unans = n - n_ans

    # single-list sampling + coverage-based filtering
    document_ids_all = [docid for docid in run[qid]]
    document_ids_filtered = []
    for i, docid in enumerate(run[qid]):
        rating = (judge[qid][docid] or [0])
        c = sum(np.array(rating) >= tau) / n_ans if n_ans > 0 else 0
        if c == 0:
            document_ids_filtered.append(docid)

    filterd_top20 = [docid for docid in document_ids_all[:20] if docid not in document_ids_filtered]
    if len(filterd_top20) > 0:
        dataset_dict['cov.pos_20.neg_51.f'].append({
            'query_id': qid, 
            'query_text': topic[str(qid)],
            'positive_document_ids': filterd_top20,
            'negative_document_ids': document_ids_all[50:],
            'subquestions': subtopics[str(qid)],
            'answer': None,
            'source': f'crux-researchy.tau:{tau}.clueweb22-B',
        })

## Transform to dataset (the base)
## Transform to dataset (other subset)
dataset_dict['train'] = dataset_dict['cov.pos_20.neg_51.f']
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
dataset.push_to_hub("DylanJHJ/crux-researchy-cov")
