# ddp_main.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler  # We'll use torch.amp.autocast with new syntax below
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Ensure required distributed environment variables are set for single-process training
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# Import your modules (make sure they are modified to work in a distributed context)
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss_functions import KnowledgeDistillationLoss
from training.trainer import Trainer
from training.validator import Validator
from evaluation.evaluator import Evaluator
from utils.data_loader import DataLoader  # This returns a DataLoader when calling get_dataloader


def main():
    # Initialize the distributed process group (use NCCL backend for GPUs)
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"

    # Create teacher and student models and move them to the appropriate device
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)

    # Wrap student model with DDP (teacher model is used only for inference)
    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)

    # Loss function with hidden_size set to 1024 (to match teacher) moved to device
    loss_fn = KnowledgeDistillationLoss(hidden_size=1024).to(device)

    # Optimizer (only student model parameters are updated)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    # Mixed precision scaler
    scaler = GradScaler()

    # Create trainer, validator, and evaluator
    trainer = Trainer(teacher_model, student_model, loss_fn, optimizer, device)
    validator = Validator(student_model, loss_fn, device)
    # When using DDP, use student_model.module for underlying model in evaluator if necessary
    # evaluator = Evaluator(student_model.module, teacher_model.tokenizer, device)

    # --- Prepare DataLoaders ---
    data_loader_obj = DataLoader(dataset_name='wikitext')
    # Instead of setting the sampler after initialization, extract the dataset and build a new DataLoader.
    base_train_loader = data_loader_obj.get_dataloader(split='train', batch_size=256)
    train_dataset = base_train_loader.dataset
    train_sampler = DistributedSampler(train_dataset)
    train_loader = TorchDataLoader(train_dataset, batch_size=256, sampler=train_sampler)

    # For validation, you can use the standard DataLoader.
    val_loader = DataLoader(dataset_name='wikitext').get_dataloader(split='validation', batch_size=256)
    # eval_loader = DataLoader(dataset_name='squad').get_dataloader(split='validation', batch_size=16)

    num_epochs = 10
    # Containers for metrics for visualization
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Main training loop
    for epoch in range(num_epochs):
        student_model.train()
        total_train_loss = 0.0
        # Set epoch for sampler for reproducible shuffling
        train_sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                    unit="batch")

        for i, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Use the updated autocast syntax.
            with torch.amp.autocast("cuda"):
                loss = trainer.train_step(input_ids, attention_mask, labels)
            # Now loss is a tensor, so we can scale and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix({
                'Iter': f"{i + 1}/{len(train_loader)}",
                'Loss': f"{loss.item():.4f}"
            })

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step (assume validator.validate returns (loss, accuracy))
        val_loss, val_accuracy = validator.validate(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Only the main process (local_rank==0) prints and saves checkpoints
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy:.2f}%")

            # Save checkpoint (student model state, optimizer state, and metrics)
            checkpoint = {
                'epoch': epoch,
                'student_model_state_dict': student_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }
            torch.save(checkpoint, "checkpoint.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}\n")

    # After training, perform evaluation on SQuAD
    # predictions = evaluator.evaluate(eval_loader)
    if local_rank == 0:
        torch.save(student_model.module.state_dict(), 'student_model.pth')

    # Visualization of training metrics (only on main process)
    # if local_rank == 0:
    #     plot_metrics(train_losses, val_losses, val_accuracies)

    # Cleanup the distributed process group
    dist.destroy_process_group()


# def plot_metrics(train_losses, val_losses, val_accuracies):
#     epochs = range(1, len(train_losses) + 1)
#     plt.figure(figsize=(14, 6))
#     # Plot losses
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_losses, label='Train Loss', marker='o')
#     plt.plot(epochs, val_losses, label='Val Loss', marker='o')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss over Epochs')
#     plt.legend()
#     # Plot accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Validation Accuracy over Epochs')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    main()
