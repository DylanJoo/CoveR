import argparse
import numpy as np
from crux.tools import load_ratings, load_run_or_qrel
from crux.tools.researchy.ir_utils import load_topic, load_subtopics

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train', help='Dataset split to analyze')
parser.add_argument('--tau', type=int, default=3, help='Threshold for answerable subtopics')
args = parser.parse_args()

# main
split = args.split
tau = args.tau

topic = load_topic(split)
subtopics = load_subtopics(split)
run = load_run_or_qrel(f'/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-{split}-init-q.bm25+qwen3.clueweb22-b.txt')
judge = load_ratings('/exp/scale25/artifacts/crux/crux-researchy/judge/')

count = 0
report = {}
for qid in run:

    # stat1: answerable 
    n = len(subtopics[qid])
    n_ans = (np.max([judge[qid][docid] for docid in judge[qid]], 0) >= tau).sum()
    n_unans = n - n_ans

    if n_unans == 0:
        count += 1

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

    report[qid] = {
        'n_subtopics': n,
        'n_answerable': int(n_ans),
        'n_unanswerable': int(n_unans),
        'r_answerable': float(n_ans/n),
        'r_unanswerable': float(n_unans/n),
        'coverage_per_doc': coverage_per_doc,
        'coverage': coverage_accumulated,
        'high_coverage_document_ids': document_ids[0.75],
        'half_coverage_document_ids': document_ids[0.5],
        'quarter_coverage_document_ids': document_ids[0.25],
        'low_coverage_document_ids': document_ids[0.0],
        'negative_document_ids': document_ids[-1],
    }


# Print and report the average
print('Total topics:', len(report), '\nTotal fully answerable topics', count)
for key in ['n_subtopics', 'n_answerable', 'n_unanswerable']:
    values = [report[qid][key] for qid in report]
    avg_value = sum(values) / len(values)
    print(f'Average {key}: {avg_value}')

for key in ['coverage_per_doc', 'coverage']:
    for rank in range(20):
        values = [report[qid][key][rank] for qid in report]
        avg_value = sum(values) / len(values)
        print(f'Average {key} at rank {rank+1}: {avg_value}')

for key in ['high_coverage_document_ids', 'half_coverage_document_ids', 'low_coverage_document_ids', 'quarter_coverage_document_ids', 'negative_document_ids']:
    values = [len(report[qid][key]) for qid in report]
    avg_value = sum(values) / len(values)
    print(f'Average {key}: {avg_value}')
