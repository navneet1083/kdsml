# training/validator.py
import torch

class Validator:
    def __init__(self, model, loss_fn, device):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def single_gpu_validate(self, dataloader):
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

    def ensure_tensor(self, x, key_name):
        """Ensure that x is a torch.Tensor. If not, try to convert it.
        If conversion fails (e.g. because x is a string), raise an error.
        """
        if isinstance(x, torch.Tensor):
            return x
        try:
            # Attempt conversion (if x is a list/number, this may work)
            return torch.tensor(x)
        except Exception as e:
            raise ValueError(
                f"Expected '{key_name}' to be convertible to a tensor, but got {x} (type {type(x)})")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for batch in val_loader:
                # Try to get required keys; adjust as needed if using SQuAD (which might have start_positions/end_positions)
                if 'input_ids' not in batch or 'attention_mask' not in batch or 'labels' not in batch:
                    raise ValueError(
                        "Validation batch is missing one of the required keys: 'input_ids', 'attention_mask', or 'labels'.")

                # Ensure each item is a tensor
                input_ids = self.ensure_tensor(batch['input_ids'], "input_ids").to(self.device)
                attention_mask = self.ensure_tensor(batch['attention_mask'], "attention_mask").to(self.device)
                labels = self.ensure_tensor(batch['labels'], "labels").to(self.device)

                # Forward pass (this example assumes a language modeling task)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # Compute loss; note: for evaluation you may need a slightly different computation
                loss = self.loss_fn(outputs, outputs, labels)  # Adjust as needed for your task
                total_loss += loss.item()

                # (Optional) Compute accuracy if applicable:
                # For example, if your task is next-token prediction:
                # predictions = outputs.argmax(dim=-1)
                # correct = (predictions == labels).sum().item()
                # total_correct += correct
                # total_examples += labels.numel()

        avg_loss = total_loss / len(val_loader)
        # For now, accuracy is set to 0.0 if you haven't computed it
        accuracy = (total_correct / total_examples * 100) if total_examples > 0 else 0.0
        return avg_loss, accuracy
