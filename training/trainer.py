# training/trainer.py
import torch

class Trainer:
    def __init__(self, teacher_model, student_model, loss_fn, optimizer, device):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_step(self, input_ids, attention_mask, labels):
        self.student_model.train()
        self.teacher_model.eval()

        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids, attention_mask)

        student_outputs = self.student_model(input_ids)
        loss = self.loss_fn(student_outputs, teacher_outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()