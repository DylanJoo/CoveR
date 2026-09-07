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
CRUX_ROOT = (os.environ["CRUX_ROOT"] or '/exp/scale25/artifacts/crux')
run = load_run_or_qrel(f'{CRUX_ROOT}/crux-researchy/runs/run.researchy-{split}-init-q.qwen3-0.6b.clueweb22-b.txt')
judge = load_ratings(f'{CRUX_ROOT}/crux-researchy/judge/')

dataset_dict = {'pos_20.neg_51.filtered': []}
for qid in tqdm(run):

    # stat1: answerable 
    n = len(subtopics[qid])
    n_ans = (np.max([judge[qid][docid] for docid in judge[qid]], 0) >= tau).sum()
    n_unans = n - n_ans

    # stat2
    document_ids = {0.75: [], 0.5: [], 0.25: [], 0.0: [], -1: []}
    coverage_per_doc = []
    coverage_accumulated = []
    for i, docid in enumerate(run[qid]):
        rating = (judge[qid][docid] or [0])
        c = sum(np.array(rating) >= tau) / n
        coverage_per_doc.append(float(c))

        selected = [docid for j, docid in enumerate(run[qid]) if j <= i]
        rating = [judge[qid][docid] for docid in selected if judge[qid][docid]]
        try:
            c_acc = sum(np.max(rating, 0) >= tau) / n
        except:
            c_acc = 0.0
        coverage_accumulated.append(float(c_acc))

        for threshold in [0.75, 0.5, 0.25, 0.0]:
            if c >= threshold:
                document_ids[threshold].append(docid)
                if c == 0:
                    document_ids[-1].append(docid)
                break

# Rank the doumcnet by the coverage
order = np.argsort(report[qid]['cov']).tolist()[::-1]
order = list(order)
positive_document_ids = [docid for i, docid in enumerate(run[qid]) if coverages[i] > 0]
negative_document_ids = [docid for i, docid in enumerate(run[qid]) if coverages[i] == 0]
dataset_list[split].append({
    'query_id': qid, 
    'query_text': topic[str(qid)],
    'positive_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance > 0],
    'negative_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance == 0],
    'answer': None,
    'coverages': coverages,
    'source': 'msmarco-passage'
})
