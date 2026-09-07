from glob import glob
from datasets import load_from_disk, DatasetDict
from datasets import concatenate_datasets

dataset_dict = {}

# subset = 'flatten'
subset = 'flatten.pos_5.neg_1'
dataset_list = []
for i, file in enumerate(glob(f'/users/judylan1/datasets/crux-researchy-{subset}.modernbert-base.pretokenized-train-*')):
    dataset_list.append( load_from_disk(file) )
    print(i)
merged = concatenate_datasets(dataset_list)
print(merged)

dataset_dict['train'] = merged
dataset_dict['eval'] = load_from_disk(f"/users/judylan1/datasets/crux-researchy-{subset}.modernbert-base.pretokenized-eval")
print(merged[0]['passage_tokenized'][0])

tokenized_hf_dataset_dict = DatasetDict({'train': dataset_dict['train'], 'eval': dataset_dict['eval']}) # save to local
tokenized_hf_dataset_dict.push_to_hub(f"DylanJHJ/crux-researchy-{subset}-pretokenized", "modernbert-base")
