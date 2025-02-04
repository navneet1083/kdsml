import torch
import os
from tqdm import tqdm
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss_functions import KnowledgeDistillationLoss
from training.trainer import Trainer
from training.validator import Validator
from evaluation.evaluator import Evaluator
from utils.data_loader import DataLoader

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Initialize models, loss, and optimizer
teacher_model = TeacherModel().to('cuda')
student_model = StudentModel().to('cuda')
loss_fn = KnowledgeDistillationLoss().to('cuda')  # Move loss function to GPU
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

# Initialize trainer and validator
trainer = Trainer(teacher_model, student_model, loss_fn, optimizer, 'cuda')
validator = Validator(student_model, loss_fn, 'cuda')

print(f'Loading train and validation dataloader ...')

# Load and preprocess training data (WikiText-103)
train_loader = DataLoader(dataset_name='wikitext').get_dataloader(split='train', batch_size=256)
val_loader = DataLoader(dataset_name='wikitext').get_dataloader(split='validation', batch_size=256)

# Load and preprocess evaluation data (SQuAD)
# eval_loader = DataLoader(dataset_name='squad').get_dataloader(split='validation', batch_size=16)


# --- Configuration ---
num_epochs = 10
checkpoint_path = os.path.join("checkpoints", "checkpoint.pth")
device = 'cuda'

# Initialize metrics containers
train_losses = []
val_losses = []
val_accuracies = []

start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    student_model.load_state_dict(checkpoint['student_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    val_accuracies = checkpoint.get('val_accuracies', [])
    print(f"Resuming training from epoch {start_epoch}")
else:
    print(f'New run with new weights')

for epoch in range(start_epoch, num_epochs):
    student_model.train()
    total_train_loss = 0.0
    epoch_steps = len(train_loader)

    # Progress bar for batches in the epoch
    pbar = tqdm(enumerate(train_loader), total=epoch_steps, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for i, batch in pbar:
        # Ensure batch tensors are on the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Perform a training step and update loss
        train_loss = trainer.train_step(input_ids, attention_mask, labels)
        total_train_loss += train_loss

        # Update progress bar with current iteration loss
        pbar.set_postfix({
            'Iter': f"{i + 1}/{epoch_steps}",
            'Loss': f"{train_loss:.4f}"
        })

    # Compute average training loss for the epoch
    avg_train_loss = total_train_loss / epoch_steps
    train_losses.append(avg_train_loss)

    # --- Validation ---
    # We assume that validator.validate returns a tuple: (loss, accuracy)
    val_loss, val_accuracy = validator.validate(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Print epoch summary with additional metrics
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  Val Acc:    {val_accuracy:.2f}%")

    # Save checkpoint including current epoch, model, optimizer, and metric histories
    checkpoint = {
        'epoch': epoch,
        'student_model_state_dict': student_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1} to '{checkpoint_path}'\n")

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