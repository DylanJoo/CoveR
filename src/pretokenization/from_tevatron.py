<<<<<<< Updated upstream
import math
=======
>>>>>>> Stashed changes
import gc
from tqdm import tqdm
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.dataset_dev import QrelDataset
from tevatron.retriever.arguments import DataArguments
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from multiprocessing import Pool, cpu_count

if __name__ == "__main__":

    # split = 'flatten'
    split = 'flatten.pos_5.neg_1'
    new_dataset_name = f'crux-researchy-{split}.modernbert-base.pretokenized'
    dataset_name = 'DylanJHJ/crux-researchy'
    corpus_name = 'DylanJHJ/crux-researchy-corpus'

    # Load config
    data_args = DataArguments(
        exclude_title=True,
        dataset_name=dataset_name,
        dataset_split=split,
        corpus_name=corpus_name,
        train_group_size=8,
        query_max_len=32,
        passage_max_len=2048,
        eval_dataset_name='DylanJHJ/Qrels',
        eval_dataset_split='msmarco_passage.trec_dl_2019',
        eval_corpus_name='Tevatron/msmarco-passage-corpus-new',
        eval_group_size=8,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

	# Pre-tokenization
    def pretokenize_batched(example):
        # Each example has a single query string and a list of passage strings
        q_text = example["query"]
        p_texts = example["passage_group"]
        p_texts = [p for group in p_texts for p in group]

        # Tokenize query
        q_enc = tokenizer(
            q_text,
            truncation=True,
            padding=False,
            max_length=data_args.query_max_len - 1 if data_args.append_eos_token else data_args.query_max_len,
            add_special_tokens=True,
        )

        # Tokenize all passages
        p_enc = tokenizer(
            p_texts,
            truncation=True,
            padding=False,
            max_length=data_args.passage_max_len - 1 if data_args.append_eos_token else data_args.passage_max_len,
            add_special_tokens=True,
        )

        regrouped_passages = []
        idx = 0
        num_passages = [len(g) for g in example["passage_group"]]
        for n in num_passages:
            regrouped_passages.append(p_enc["input_ids"][idx:idx + n])
            idx += n
        
        return {
            "query_tokenized": q_enc["input_ids"],
            "passage_tokenized": regrouped_passages
        }


	# Load dataset
    dataset_dict = {}
    for split in ['eval', 'train']:

        # Load tevatron dataset
        if split == 'train':
            pt_dataset = TrainDataset(data_args)
        if split == 'eval':
            pt_dataset = QrelDataset(data_args, corpus_name=data_args.eval_corpus_name)

		# Convert to hf
        def convert_item(i):
            query, passages = pt_dataset[i]
            return {"query": query[0], "passage_group": [p[0] for p in passages] }

        with Pool(cpu_count()) as p:
            hf_data = list(tqdm(p.imap(convert_item, range(len(pt_dataset))), total=len(pt_dataset)))

		# Move the list
        hf_dataset = Dataset.from_list(hf_data)
		del pt_dataset
        del hf_data
        gc.collect()

		# Apply tokenization
        if split == 'eval':
            tokenized_hf_dataset = hf_dataset.map(pretokenize_batched, num_proc=4, load_from_cache_file=False, batched=True)
            tokenized_hf_dataset.save_to_disk(f"/users/judylan1/datasets/{new_dataset_name}-eval")

        if split == 'train':
            chunk_size = 100000
            total = len(hf_dataset)
            for i in range(math.ceil(total / chunk_size)):
                subset = hf_dataset.select(range(i * chunk_size, min((i+1) * chunk_size, total)))
                tokenized_hf_dataset = subset.map(pretokenize_batched, batched=True, num_proc=16)
                tokenized_hf_dataset.save_to_disk(f"/users/judylan1/datasets/{new_dataset_name}-train-{i}")
                del subset, tokenized_hf_dataset
                gc.collect()
