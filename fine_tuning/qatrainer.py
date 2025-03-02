import torch.nn as nn
import torch
import re, string


class QA_Trainer:
    def __init__(self, model, optimizer, device, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch):
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        start_positions = batch["start_positions"].to(self.device)
        end_positions = batch["end_positions"].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # If output is a dict (return_dict=True), extract logits; else assume tuple.
        if hasattr(outputs, "start_logits"):
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        else:
            start_logits, end_logits = outputs
        loss_start = self.criterion(start_logits, start_positions)
        loss_end = self.criterion(end_logits, end_positions)
        loss = (loss_start + loss_end) / 2.0
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_f1 = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "start_logits"):
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                else:
                    start_logits, end_logits = outputs
                loss_start = self.criterion(start_logits, start_positions)
                loss_end = self.criterion(end_logits, end_positions)
                loss = (loss_start + loss_end) / 2.0
                total_loss += loss.item()
                # Compute F1 score for each example.
                pred_start = start_logits.argmax(dim=-1)
                pred_end = end_logits.argmax(dim=-1)
                for i in range(len(input_ids)):
                    offsets = batch["offset_mapping"][i]  # List of [start, end] pairs
                    context = batch["context"][i]  # String
                    gt_answers = batch["answers"][i]["text"]  # List of ground truth answers
                    s_idx = pred_start[i].item()
                    e_idx = pred_end[i].item()
                    if s_idx > e_idx:
                        predicted_answer = ""
                    else:
                        start_char = offsets[s_idx][0]
                        end_char = offsets[e_idx][1]
                        predicted_answer = context[start_char:end_char]
                    f1 = 0.0
                    for gt in gt_answers:
                        f1 = max(f1, compute_f1(predicted_answer, gt))
                    total_f1 += f1
                    count += 1
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        avg_f1 = total_f1 / count if count > 0 else 0.0
        return avg_loss, avg_f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    """Compute the F1 score between a prediction and a ground truth answer."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1