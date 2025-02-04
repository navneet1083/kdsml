# utils/data_loader.py
from datasets import load_dataset
from transformers import BertTokenizer
import torch

class DataLoader:
    def __init__(self, dataset_name='wikitext', tokenizer_name='bert-large-uncased', max_length=128):
        self.dataset = load_dataset(dataset_name, 'wikitext-103-v1')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def preprocess(self, examples):
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def get_dataloader(self, split='train', batch_size=32):
        dataset = self.dataset[split].map(self.preprocess, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))