# utils/data_loader.py
from datasets import load_dataset
from transformers import BertTokenizer
import torch
import os


class DataLoader:
    def __init__(self, dataset_name='wikitext', tokenizer_name='bert-large-uncased', max_length=128):
        self.dataset_name = dataset_name
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def _preprocess_wikitext(self, examples):
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def preprocess_wikitext(self, examples):
        tokenized = self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Create labels by shifting input_ids by one position
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift left
        labels[:, -1] = self.tokenizer.pad_token_id  # Pad the last token

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def preprocess_squad(self, examples):
        # Tokenize question and context
        inputs = self.tokenizer(
            examples['question'],
            examples['context'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Extract answer start and end positions
        start_positions = []
        end_positions = []
        for answer in examples['answers']:
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])

            # Map character positions to token positions
            sequence_ids = inputs.sequence_ids(0)
            char_to_token = lambda idx: inputs.char_to_token(idx)

            start_token = char_to_token(start_char)
            end_token = char_to_token(end_char - 1)

            # Handle cases where answer is truncated
            if start_token is None or end_token is None:
                start_token, end_token = 0, 0
            start_positions.append(start_token)
            end_positions.append(end_token)

        inputs['start_positions'] = torch.tensor(start_positions)
        inputs['end_positions'] = torch.tensor(end_positions)
        return inputs

    def get_dataloader(self, split='train', batch_size=32):
        if self.dataset_name == 'wikitext':
            # Load WikiText-103 dataset
            dataset = load_dataset('wikitext', 'wikitext-103-v1')
            dataset = dataset[split].map(self.preprocess_wikitext, batched=True, num_proc=os.cpu_count())
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        elif self.dataset_name == 'squad':
            # Load SQuAD dataset
            dataset = load_dataset('squad', split=split)
            dataset = dataset.map(self.preprocess_squad, batched=True, num_proc=os.cpu_count())
            dataset.set_format(type='torch',
                               columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))