
class FineTuneTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch):
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item()
        return loss.item(), acc

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == labels).float().mean().item()
                total_loss += loss.item()
                total_acc += acc
                count += 1
        return total_loss / count, total_acc / count