"""
crux-researchy
split='flatten.pos_5.neg_1': positive from document >= 5; negatives from document <= 1
"""
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
parser.add_argument('--split_flatten', type=str, help='the name for the new split')
args = parser.parse_args()

# main
split = args.split
tau = args.tau

topic = load_topic(split)
subtopics = load_subtopics(split)
CRUX_ROOT = (os.environ["CRUX_ROOT"] or '/exp/scale25/artifacts/crux')
run = load_run_or_qrel(f'{CRUX_ROOT}/crux-researchy/runs/run.researchy-{split}-init-q.bm25+qwen3.clueweb22-b.txt')
judge = load_ratings(f'{CRUX_ROOT}/crux-researchy/judge/')

dataset_dict = {'flatten': []}
for qid in tqdm(run):

    # stat1: answerable 
    n = len(subtopics[qid])
    n_ans = (np.max([judge[qid][docid] for docid in judge[qid]], 0) >= tau).sum()

    # 3. flatten all the sub-questions
    for i, subtopic in enumerate(subtopics[qid]):
        
        positive_docs = [docid for docid in judge[qid] if judge[qid][docid][i] >= tau]
        negative_docs = [docid for docid in judge[qid] if judge[qid][docid][i] <= 1]

        if len(positive_docs) >= 1:
            negative_docs_unjudged = [docid for docid in run[qid]][51:(67 - len(negative_docs))]
            negative_docs += negative_docs_unjudged # add unjudged documents

            dataset_dict['flatten'].append({
                'query_id': f"{qid}#{i}",
                'query_text': subtopic,
                'positive_document_ids': positive_docs,
                'negative_document_ids': negative_docs,
                'answer': None,
                'source': f'crux-researchy.tau:{tau}.clueweb22-B:{topic[qid]}',
            })

## Transform to dataset (the base)
## Transform to dataset (other subset)
dataset = Dataset.from_list(dataset_dict['flatten'])
print(dataset)
dataset.push_to_hub("DylanJHJ/crux-researchy", split=args.split_flatten)
