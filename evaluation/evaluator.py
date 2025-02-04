# evaluation/evaluator.py
import torch

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, dataset):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for example in dataset:
                inputs = self.tokenizer(example['question'], example['context'], return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                predictions.append(outputs.argmax(dim=-1).cpu().numpy())

        return predictions