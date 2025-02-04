# main.py
import torch
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss_functions import KnowledgeDistillationLoss
from training.trainer import Trainer
from training.validator import Validator
from evaluation.evaluator import Evaluator
from utils.data_loader import DataLoader

# Initialize models, loss, and optimizer
teacher_model = TeacherModel().to('cuda')
student_model = StudentModel().to('cuda')
loss_fn = KnowledgeDistillationLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

# Initialize trainer and validator
trainer = Trainer(teacher_model, student_model, loss_fn, optimizer, 'cuda')
validator = Validator(student_model, loss_fn, 'cuda')

# Load and preprocess training data (WikiText-103)
train_loader = DataLoader(dataset_name='wikitext').get_dataloader(split='train', batch_size=32)
val_loader = DataLoader(dataset_name='wikitext').get_dataloader(split='validation', batch_size=32)

# Load and preprocess evaluation data (SQuAD)
eval_loader = DataLoader(dataset_name='squad').get_dataloader(split='validation', batch_size=16)

# Training loop
train_losses, val_losses = [], []
for epoch in range(10):
    # Train on WikiText-103
    total_train_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask = batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda')
        labels = input_ids.clone()  # Self-supervised learning: predict masked tokens
        train_loss = trainer.train_step(input_ids, attention_mask, labels)
        total_train_loss += train_loss
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validate on WikiText-103
    val_loss = validator.validate(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss}, Val Loss = {val_loss}")

# Evaluate on SQuAD
evaluator = Evaluator(student_model, teacher_model.tokenizer, 'cuda')
predictions = evaluator.evaluate(eval_loader)

# Save student model weights
torch.save(student_model.state_dict(), 'student_model.pth')

# Plot metrics
# def plot_metrics(train_losses, val_losses):
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Training and Validation Loss')
#     plt.show()
#
# plot_metrics(train_losses, val_losses)