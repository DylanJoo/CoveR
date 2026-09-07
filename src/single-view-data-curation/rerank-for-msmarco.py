""" 
The reranking pipeline is copied from Qwen3-Reranker-0.6B on Huggingface 
NOTE: revise if needed. I accidentally remove the reranking codes. but it's fine i have already secured them :(
Revision is based on this code.
"""
from typing import Dict, Optional, List

import json
import logging

import torch

from transformers import AutoTokenizer, is_torch_npu_available
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from vllm.inputs.data import TokensPrompt

import os
import argparse
from tqdm import tqdm

def format_instruction(instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(tokenizer, pairs, instruction, max_length, suffix_tokens):
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages =  tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        token_count = len(outputs[i].outputs[0].token_ids)
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores

def rerank(batch_size, split='train'):

    # prepare model and tokenizer
    number_of_gpu = torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B')
    model = LLM(model='Qwen/Qwen3-Reranker-0.6B', tensor_parallel_size=number_of_gpu, max_model_len=10240, enable_prefix_caching=True, gpu_memory_utilization=0.9)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    max_length=8192
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(temperature=0, 
        max_tokens=1,
        logprobs=20, 
        allowed_token_ids=[true_token, false_token],
    )

    # prepare corpus and runs
    from crux.tools import load_run_or_qrel, batch_iterator
    from datasets import load_dataset

    run = load_run_or_qrel('/users/judylan1/datasets/msmarco-passage-kd/run.msmarco-v1-passage.bm25+qwen3-default.train.txt')

    ds = load_dataset('Tevatron/msmarco-passage-new')['train']
    queries = {ex['query_id']: ex['query_text'] for ex in ds}
    print(f"Data loading complete. Number of queries: {len(queries)}")

    candidates = {ex['query_id']: ex['positive_document_ids'] + ex['negative_document_ids'] for ex in ds}

    ds = load_dataset('Tevatron/msmarco-passage-corpus-new')['train']
    corpus = {ex['docid']: ex['text'] for ex in ds}
    print(f"Data loading complete. Number of documents: {len(corpus)}")

    output_run = f'/users/judylan1/datasets/msmarco-passage-kd/run.tevatron-msmarco-passage-new.train.txt'
    if os.path.exists(output_run):
        run_done = load_run_or_qrel(output_run)
        output_run += ".new"

    ## prepare multiquery
    task = 'Given the list of questions as query, retrieve relevant passages that answer the questions.'

    ## Get documents to be judged
    document_ids = {}
    documents = {}
    for qid in queries:
        # ignore the one in run file 
        candidate_docids = [docid for docid in candidates[qid] if docid not in run[qid]]
        document_ids[qid] = [docid for docid in candidate_docids if docid not in run_done[qid]]
        documents[qid] = [corpus[docid] for docid in document_ids[qid]]

    ## Pairs of ids and query-document
    all_scores = []
    for qid in tqdm(queries, desc=f"Processing queries"):
        query = queries[qid]
        if len(documents[qid]) > 0:
            pairs = [(query, " ".join(document.split()[:5000])) for document in documents[qid]]
            inputs = process_inputs(tokenizer, pairs, task, max_length-len(suffix_tokens), suffix_tokens)
            scores = compute_logits(model, inputs, sampling_params, true_token, false_token)

            # sort and covert score into run file
            with open(output_run, 'a') as f:
                docid_score_pairs = sorted(zip(document_ids[qid], scores), key=lambda x: x[1], reverse=True)
                for rank, (docid, score) in enumerate(docid_score_pairs, start=1):
                    f.write(f"{qid} Q0 {docid} {100+rank} {score:.6f} BM25+qwen3-0.6b\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--split', type=str, default='train', help='data split, train or dev')
    args = parser.parse_args()

    rerank(batch_size=args.batch_size, split=args.split)
    destroy_model_parallel()
