import torch.nn as nn
import torch


class QA_Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch):
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        start_positions = batch["start_positions"].to(self.device)
        end_positions = batch["end_positions"].to(self.device)
        self.optimizer.zero_grad()
        start_logits, end_logits = self.model(input_ids, attention_mask)
        loss_start = self.criterion(start_logits, start_positions)
        loss_end = self.criterion(end_logits, end_positions)
        loss = (loss_start + loss_end) / 2.0
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)
                start_logits, end_logits = self.model(input_ids, attention_mask)
                loss_start = self.criterion(start_logits, start_positions)
                loss_end = self.criterion(end_logits, end_positions)
                loss = (loss_start + loss_end) / 2.0
                total_loss += loss.item()
                count += 1
        return total_loss / count