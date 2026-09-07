from tqdm import tqdm
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.dataset_dev import QrelDataset
from tevatron.retriever.arguments import DataArguments
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from multiprocessing import Pool, cpu_count

new_dataset_name = 'msmarco-passage-new-pretokenized'
dataset_name = 'Tevatron/msmarco-passage-new'
corpus_name = 'Tevatron/msmarco-passage-corpus-new'
split = 'train'

# Load config
data_args = DataArguments(
    exclude_title=True,
    dataset_name=dataset_name,
    dataset_split=split,
    corpus_name=corpus_name,
    train_group_size=8,
    query_max_len=32,
    passage_max_len=512,
    eval_dataset_name='DylanJHJ/Qrels',
    eval_dataset_split='msmarco_passage.trec_dl_2019',
    eval_corpus_name='Tevatron/msmarco-passage-corpus-new',
    eval_group_size=8,
)

# Pre-tokenization
def pretokenize(example):
    # Each example has a single query string and a list of passage strings
    q_text = example["query"]
    p_texts = example["passage_group"]

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

    return {
        "query_tokenized": q_enc["input_ids"],
        "passage_tokenized": p_enc["input_ids"],
    }


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')

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

    # Apply tokenization
    tokenized_hf_dataset = hf_dataset.map(pretokenize, num_proc=32)
    dataset_dict[split] = tokenized_hf_dataset

# Save for later use
tokenized_hf_dataset_dict = DatasetDict( 
    {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict}
)
tokenized_hf_dataset_dict.save_to_disk(f"/scratch-shared/dju/datasets/{new_dataset_name}")
