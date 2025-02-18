import torch

class Validator:
    def __init__(self, student_model, teacher_model, loss_fn, device):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.loss_fn = loss_fn
        self.device = device

    def ensure_tensor(self, x, key_name):
        if isinstance(x, torch.Tensor):
            return x
        try:
            return torch.tensor(x)
        except Exception as e:
            raise ValueError(f"Expected '{key_name}' to be convertible to a tensor, but got {x} (type {type(x)})") from e

    def validate(self, val_loader, max_batches=None):
        self.student_model.eval()
        self.teacher_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_count += 1
                # Ensure required keys exist
                if 'input_ids' not in batch or 'attention_mask' not in batch or 'labels' not in batch:
                    raise ValueError("Validation batch is missing one of: 'input_ids', 'attention_mask', or 'labels'.")

                input_ids = self.ensure_tensor(batch['input_ids'], "input_ids").to(self.device)
                attention_mask = self.ensure_tensor(batch['attention_mask'], "attention_mask").to(self.device)
                labels = self.ensure_tensor(batch['labels'], "labels").to(self.device)

                # Get teacher hidden states from the teacher model
                teacher_outputs = self.teacher_model(input_ids, attention_mask=attention_mask)
                teacher_hidden_states = teacher_outputs['last_hidden_state']  # Expected shape: (batch, seq, 1024)

                # Get student outputs
                student_outputs = self.student_model(input_ids, attention_mask=attention_mask)  # Shape: (batch, seq, vocab_size)

                # Compute loss using the teacher hidden states
                loss = self.loss_fn(student_outputs, teacher_hidden_states, labels)
                total_loss += loss.item()

                # (Optional) Compute accuracy if applicable:
                # predictions = student_outputs.argmax(dim=-1)
                # correct = (predictions == labels).sum().item()
                # total_correct += correct
                # total_examples += labels.numel()

                if max_batches is not None and batch_count >= max_batches:
                    break

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        # Here accuracy is set to 0.0 if not computed
        accuracy = (total_correct / total_examples * 100) if total_examples > 0 else 0.0
        return avg_loss, accuracy
