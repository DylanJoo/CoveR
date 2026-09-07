import gzip
import json
import os
from glob import glob
from tqdm import tqdm
from datasets import Dataset
from crux.tools import load_run_or_qrel

dataset = []
home_dir = os.path.expanduser("~/")

## QRELS from TREC
qrel = load_run_or_qrel("/home/dju/datasets/crux/crux-neuclir/qrels/neuclir24-test-request.qrel")
run = load_run_or_qrel(f"runs/qwen3-8B-embed.neuclir.txt", topk=100)
all_docids1 = {docid for qid in run for docid in run[qid].keys()}
run = load_run_or_qrel(f"runs/bm25.neuclir.txt", topk=100)
all_docids2 = {docid for qid in run for docid in run[qid].keys()}
all_docids = all_docids1 | all_docids2

files = glob('/home/dju/datasets/neuclir1/*jsonl.gz')

document = {}
for file in files:
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line)
            docid = item.pop('id')
            if docid in all_docids:
                document[docid] = item
                all_docids.remove(docid)
print('Total docs:', len(document))

for docid in document:
    dataset.append({
        'docid': docid,
        'title': document[docid]["title"],
        'text': document[docid]["text"],
        'source': 'qwen3-8B-top100 | bm25-top100',
    })

## Transform to dataset
dataset = Dataset.from_list(dataset)
dataset.push_to_hub("DylanJHJ/neuclir1-subset-corpus")
