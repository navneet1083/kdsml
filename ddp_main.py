import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler  # using new autocast syntax below
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Ensure required distributed environment variables are set for single-process training
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss_functions import KnowledgeDistillationLoss
from training.trainer import Trainer
from training.validator import Validator
from evaluation.evaluator import Evaluator
from utils.data_loader import DataLoader  # Returns a DataLoader from get_dataloader


def main():
    # Initialize the distributed process group
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    print('--'*20)
    print(f'Local Rank: {local_rank} :: device : {device}')
    print('--'*20)

    # Create teacher and student models and move them to device
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)
    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)

    # Loss function with hidden_size set to 1024 to match teacher; move to device
    loss_fn = KnowledgeDistillationLoss(hidden_size=1024).to(device)

    # Optimizer (only updating student model parameters)
    # optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    # Mixed precision scaler
    scaler = GradScaler()

    # Create trainer, validator (pass teacher model), and evaluator.
    trainer = Trainer(teacher_model, student_model, loss_fn, optimizer, device)
    validator = Validator(student_model, teacher_model, loss_fn, device)
    # evaluator = Evaluator(student_model.module, teacher_model.tokenizer, device)

    # --- Check for existing checkpoint ---
    checkpoint_path = "checkpoint.pth"
    start_epoch = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student_model.module.load_state_dict(checkpoint['student_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        if local_rank == 0:
            print(f"Resuming training from epoch {start_epoch}.")

    # --- Prepare DataLoaders ---
    data_loader_obj = DataLoader(dataset_name='wikitext')
    base_train_loader = data_loader_obj.get_dataloader(split='train', batch_size=256)
    train_dataset = base_train_loader.dataset
    train_sampler = DistributedSampler(train_dataset)
    train_loader = TorchDataLoader(train_dataset, batch_size=256, sampler=train_sampler)

    val_loader = DataLoader(dataset_name='wikitext').get_dataloader(split='validation', batch_size=256)
    # eval_loader = DataLoader(dataset_name='squad').get_dataloader(split='validation', batch_size=256)

    num_epochs = 50
    eval_interval = 100  # mini-evaluation every 100 batches

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        student_model.train()
        total_train_loss = 0.0
        train_sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                    unit="batch")

        for i, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = trainer.train_step(input_ids, attention_mask, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix({
                'Iter': f"{i + 1}/{len(train_loader)}",
                'Loss': f"{loss.item():.4f}"
            })

            # Mini evaluation check every eval_interval iterations (only on main process)
            # if (i + 1) % eval_interval == 0 and local_rank == 0:
            #     print("\nPerforming mini evaluation check...")
            #     try:
            #         mini_val_loss, mini_val_acc = validator.validate(val_loader, max_batches=1)
            #         print(f"Mini evaluation: Loss = {mini_val_loss:.4f}, Accuracy = {mini_val_acc:.2f}%")
            #     except Exception as e:
            #         print("Mini evaluation failed:", e)
            #         raise e

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Full validation at end of epoch
        val_loss, val_accuracy = validator.validate(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if local_rank == 0:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy:.2f}%")
            # Save checkpoint after each epoch
            checkpoint = {
                'epoch': epoch,
                'student_model_state_dict': student_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch + 1}\n")

    # predictions = evaluator.evaluate(eval_loader)
    if local_rank == 0:
        torch.save(student_model.module.state_dict(), 'student_model.pth')
        save_training_plot(train_losses, val_losses, val_accuracies)
    dist.destroy_process_group()


def save_training_plot(train_losses, val_losses, val_accuracies):
    """Save the training metrics plot to a file in the 'plots' directory using a descriptive naming convention."""
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    # Create a timestamp for the file name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use a naming convention: training_metrics_epoch{last_epoch}_{timestamp}.png
    filename = f"plots/training_metrics_epoch{len(train_losses)}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Training metrics plot saved as: {filename}")


if __name__ == "__main__":
    main()
