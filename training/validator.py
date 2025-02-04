# training/validator.py
import torch

class Validator:
    def __init__(self, model, loss_fn, device):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

                outputs = self.model(input_ids)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)